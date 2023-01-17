import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
gender = pd.read_csv('../input/titanic/gender_submission.csv')

test = pd.read_csv('../input/titanic/test.csv')

train = pd.read_csv('../input/titanic/train.csv')
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']

train = train.drop(drop_elements, axis = 1)

colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

train.Sex = train.Sex.map({'female':1, 'male':0}) 

test.Sex = test.Sex.map({'female':1, 'male':0})

#sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            #square=True, cmap=colormap, linecolor='white', annot=True)
avg_fare = test['Fare'].mean()

test['Fare'] = test['Fare'].fillna(avg_fare)
X_train = train[['Sex','Fare','Pclass']].values

Y_train = train[['Survived']].values
#scaler = StandardScaler()

#X_train = scaler.fit_transform(X_train)

clf = SVC(gamma='auto')

clf.fit(X_train, Y_train)
train_predict = clf.predict(X_train)

accuracy_score(Y_train, train_predict)
X_test = test[['Sex','Fare','Pclass']].values

#X_test_scaled = scaler.transform(X_test)



Y_test = gender[['Survived']].values
test_predict = clf.predict(X_test)

accuracy_score(Y_test, test_predict)
from sklearn.ensemble import RandomForestClassifier



random_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

random_clf.fit(X_train, Y_train)

random_predict = random_clf.predict(X_train)

accuracy_score(Y_train, random_predict)
test_prediction = random_clf.predict(X_test)

accuracy_score(Y_test, test_prediction)