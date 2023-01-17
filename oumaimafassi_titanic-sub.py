from sklearn.preprocessing import *

import pandas as pd 

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
# Import Data

df_train =  pd.read_csv('../input/train.csv')

df_sub = pd.read_csv('../input/test.csv')

df_sub['Survived'] = -99
df_sub['Survived']
# preprocessing 

df = df_train.append(df_sub)



df['Age'].fillna( df['Age'].mean() , inplace=True)

df['Fare'].fillna( 0 , inplace=True)

df['title'] = df['Name'].apply( lambda x : x.split(',')[1].split('.')[0])

df['title'] = LabelEncoder().fit_transform(df['title'])

df['Sex'] = LabelEncoder().fit_transform( df['Sex'] )

df['Embarked'].fillna( 'unknown' , inplace=True)

df['Embarked'] = LabelEncoder().fit_transform( df['Embarked'] )

df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)



df.columns

X = df.loc[df['Survived']!=-99][['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp','title']]

y = df.loc[df['Survived']!=-99]['Survived']





sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)



clf = RandomForestClassifier(n_estimators=20, random_state=0)



parameters_dc = { 'max_depth' : range(3,10),

                  'min_samples_split': [2, 5, 10, 20]}



gs = GridSearchCV(estimator=clf, param_grid=parameters_dc, cv=sk )

gs.fit(X , y  )

print ( gs.best_params_ )

print ("ERROR : ", 1-gs.best_score_)
model = RandomForestClassifier(n_estimators=20, random_state=0, max_depth= 9, min_samples_split= 2)

model.fit(X,y)
X_test = df.loc[df['Survived']==-99][['Age', 'Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'SibSp','title']]

model.predict(X_test)
df_submission = pd.concat( [df_sub['PassengerId'], pd.Series(list(model.predict(X_test)))], axis=1 )

#



df_submission.columns=['PassengerId', 'Survived']

df_submission

df_submission.to_csv('submission.csv', index = False)