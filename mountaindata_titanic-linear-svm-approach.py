

import pandas as pd

from pandas import Series,DataFrame

import numpy as np

from sklearn import preprocessing

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import hinge_loss



#Import Data, remove impertinent data fields, Cabin might have been useful but it has so many NaN values

df = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df= df.drop(['Name','Ticket','PassengerId','Embarked', 'Cabin'],axis = 1)

df_test = df_test.drop(['Name','Ticket','Embarked', 'Cabin'],axis = 1)



#Clean Data, age is often empty, provide an average if empty

df['Gender'] = df['Sex'].map( {'female' : 0, 'male' : 1}).astype(int)

df = df.drop(['Sex'], axis = 1)

df = df.fillna(df.mean())



df_test['Gender'] = df_test['Sex'].map( {'female' : 0, 'male' : 1}).astype(int)

df_test = df_test.drop(['Sex'], axis = 1)

df_test = df_test.fillna(df.mean())



#Declare Training and test data

train_data = df.drop(['Survived'],axis = 1)

train_labels = df.Survived

test_data = df_test.drop(['PassengerId'], axis = 1)



#Train and Test Linear SVM Model

svc = SVC(kernel='linear', C=1)

svc.fit(train_data, train_labels)

predicted = svc.predict(test_data)



#Create Submission dataframe and save to csv

submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": predicted

    })



submission.to_csv('submission.csv', index = False)