import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic.head()
titanic.shape
titanic.isnull().sum()
titanic.dtypes
titanic.nunique()
#Dropping Name as it should not impact survival

#Dropping Ticket as it has nearly 80% unique values and would not help in prediction

#Dropping Cabin as it has neaarly 80% missing values



titanic.drop(['Name', 'Ticket','Cabin'], axis=1, inplace=True)

titanic.head()
titanic['Embarked'].unique()
#Encoding the categorical data



titanic.replace({'male': 1, 'female': 0}, inplace=True)

titanic.replace({'S':1, 'C':2, 'Q':3}, inplace=True)
#Filling missing values



titanic.interpolate(method='linear', limit_direction='both', inplace=True)
titanic.isnull().sum()
ax = plt.subplots(figsize=(10,5))

ax = sns.heatmap(titanic.corr(), annot=True, linewidths=0.5, cmap='YlGnBu', fmt='.1f')
titanic.describe()
x = titanic.iloc[:,2:]

y = titanic.iloc[:,1]
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFECV

from sklearn.model_selection import StratifiedKFold
rfc = RandomForestClassifier(random_state=0, n_estimators=120)

rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')

rfecv.fit(x,y)
plt.figure(figsize=(10, 5))

plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)

plt.xlabel('Number of features selected', fontsize=14, labelpad=20)

plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)



plt.show()
print('Optimal number of features: {}'.format(rfecv.n_features_))
x.drop(x.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
x.shape
x.columns
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
lr = LogisticRegression(random_state=0, solver='liblinear')

dt = DecisionTreeClassifier(max_depth=2, random_state=0)

rfc = RandomForestClassifier(n_estimators=120, random_state=0)

knn = KNeighborsClassifier(n_neighbors=2)

svc = SVC(kernel='rbf', random_state=0, gamma='auto')

gnb = GaussianNB()
models = [lr, dt, rfc, knn, svc, gnb]

for model in models:

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    print(model, 'model accuracy is ', accuracy_score(y_test,y_pred))
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')

titanic_test.head()
titanic_test.drop(['Name','Ticket','Cabin', 'Embarked'], axis=1, inplace=True)

titanic_test.head()
titanic_test.replace({'male':1, 'female':0}, inplace=True)
titanic_test.dtypes
titanic_test.shape
titanic_test.isnull().sum()
titanic_test.interpolate(method='linear', limit_direction='both', inplace=True)
titanic_test.isnull().sum()
x_final_pred = titanic_test.iloc[:, 1:]



final_pred = rfc.predict(x_final_pred)
output = pd.DataFrame(titanic_test['PassengerId'])

output['Survived'] = final_pred

output.head()
output.to_csv('my_submission_subho.csv')