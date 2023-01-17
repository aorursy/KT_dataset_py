import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
titanictrain = pd.read_csv("../input/train.csv")
titanictrain.head()
sex = pd.get_dummies(titanictrain['Sex'],drop_first=True)

embark = pd.get_dummies(titanictrain['Embarked'],drop_first=True)
titanictrain.drop(['Sex','Embarked'],axis=1,inplace=True)

train = pd.concat([titanictrain,sex,embark],axis=1)
train.head()
import re



def splitter(cols):

    return re.split('\, |\. ',cols)[1]





train['Name'] = train['Name'].apply(splitter)
train.head()
def age_imputation(cols):

    

    Age=cols[0]

    Name=cols[1]

    

    if pd.isnull(Age):

        if Name == 'Mr':

            return 32

        elif Name == 'Mrs':

            return 36

        elif Name == 'Miss':

            return 22

        elif Name == 'Master':

            return 5

        elif Name == 'Dr':

            return 42

    else:

        return Age
train['Age'] = train[['Age','Name']].apply(age_imputation,axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
X = train.drop('Survived',axis=1)

y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(rfc_pred,y_test))