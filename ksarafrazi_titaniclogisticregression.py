import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.linear_model import LogisticRegression
#Loading the training data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#imputing the missing age values
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
def impute_fare(cols):
    Fare = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Fare):

        if Pclass == 1:
            return 84

        elif Pclass == 2:
            return 20

        else:
            return 13

    else:
        return Fare
#Dealing with the missing values
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)

#Dealing with the categorical data
#Encoding sex and embarked columns
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
#Dropping useless columns 
train.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
#Dealing with the missing values
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
test['Fare'] = test[['Fare','Pclass']].apply(impute_fare,axis=1)

#Encoding sex and embarked columns
sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
#Dropping useless columns 

test.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
test = pd.concat([test,sex,embark],axis=1)

x = train.drop('Survived',axis=1)
y= train['Survived']

logmodel = LogisticRegression()
logmodel.fit(x,y)

predictions = logmodel.predict(test)

submission = pd.read_csv('../input/gender_submission.csv')
submission ['Survived']= predictions
submission.to_csv('submission.csv', index=False)


submission.shape
