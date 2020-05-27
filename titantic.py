import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def impute_age(cols):
    age=cols[0]
    pClass = cols[1]
    if pd.isnull(age):
        if pClass == 1:
            return 37
        elif pClass == 2:
            return 29
        else:
            return 24
    else:
        return age

train_df = pd.read_csv('titanic_train.csv')
train_df['Age'] = train_df[['Age','Pclass']].apply(impute_age,axis=1)
gender = pd.get_dummies(train_df['Sex'],drop_first=True)
ebarked = pd.get_dummies(train_df['Embarked'],drop_first=True)
train_df = pd.concat([train_df,gender,ebarked],axis=1)
train_df.drop(['Sex','Embarked'],axis=1,inplace=True)
train_df.drop('PassengerId',axis=1,inplace=True)
train_df.drop('Name',axis=1,inplace=True)
train_df.drop('Ticket',axis=1,inplace=True)
train_df.drop('Cabin',axis=1,inplace=True)
X = train_df.drop('Survived',axis=1)
y = train_df['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=102)
joblib_fileName = "titance.pkl"
jonlib_Titantic_Model = joblib.load(joblib_fileName)
predict = jonlib_Titantic_Model.predict(X_test)
print("############ Classification Report ############")
print(classification_report(y_test,predict))
print("############ Confusion Matrix ############")
print(confusion_matrix(y_test,predict))