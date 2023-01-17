# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

df.head()

df.info()
df.describe()
fig, axes = plt.subplots(figsize=(10, 6))

sns.heatmap(df.corr(), annot=True, cmap="magma")

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

sns.distplot(df['CGPA'])

plt.title('Distribution plot for CGPA')
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

sns.distplot(df['GRE Score'])

plt.title('Distribution plot for GRE')
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

sns.distplot(df['TOEFL Score'])

plt.title('Distribution plot for TOFEL score')
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

plt.scatter(x= df['TOEFL Score'], y=df['GRE Score'])

plt.title('TOFEL Score V/s GRE Score')
sns.countplot(x='Research', data=df)
for i in df[df['Chance of Admit ']>0.75]:

    print(i)
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split



y = df['Chance of Admit ']

x = df.drop('Chance of Admit ', axis=1)



x_train, x_test,y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

'''y_train_01 = [1 if each > 0.8 else 0 for each in y_train]

y_test_01 = [1 if each > 0.8 else 0 for each in y_train]'''



y_train[y_train>=0.8] = 1

y_train[y_train<0.8] = 0



y_test[y_test>=0.8] = 1

y_test[y_test<0.8] = 0
x_train.drop('Serial No.', axis=1, inplace=True)

x_test.drop('Serial No.', axis=1, inplace=True)

print('Serial No. dropped')
columns = x.columns

print(columns.values)

x['GRE Score'].unique()
plt.hist('GRE Score', data= x)
plt.hist('TOEFL Score', data= x)
plt.hist('University Rating', data= x)
x['SOP'].unique()
plt.hist('SOP', data= x)
x['LOR '].unique()
plt.hist('LOR ', data= x)
plt.hist('CGPA', data= x)
plt.hist('Research', data= x)
x['Research'].unique()
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import make_scorer,f1_score, accuracy_score, precision_score



print(x_test.shape)

print(y_test.shape)

x_test.head()
y_test.head()
import warnings

warnings.simplefilter(action='ignore')



acc_scorer = make_scorer(accuracy_score)



accuracy_scoring  = {}

f1_scoring = {}

params = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}

model = GridSearchCV(LogisticRegression(), param_grid = params, scoring = acc_scorer, cv=5)



model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Logistics Regression')

print('Best Parameter', model.best_params_)

print('Training Accuracy Score', model.best_score_)



v1 = accuracy_score(y_test, pred)

v2 = f1_score(y_test, pred)

print('Testing Accuracy Score', v1)

print('Testing f1 Score', v2)

accuracy_scoring['LogisticRegression Testing Accuracy Score'] = v1

f1_scoring['LogisticRegression Testing f1 Score'] = v2
params = {"n_estimators": [i for i in range(1, 200, 10)]}

print('RandomForest Classifier')

model = GridSearchCV(RandomForestClassifier(), param_grid = params, scoring = acc_scorer, cv=5)



model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Best Parameter', model.best_params_)

print('Training Accuracy Score', model.best_score_)



v3 = accuracy_score(y_test, pred)

v4 = f1_score(y_test, pred)

print('Testing Accuracy Score', v3)

print('Testing f1 Score', v4)

accuracy_scoring['RandomForestClassifier Testing Accuracy Score'] = v3

f1_scoring['RandomForestClassifier Testing f1 Score'] = v4
print('Support Vector Machine')

param_grid = {

    'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],

    'kernel': ['linear', 'rbf']

}

model = GridSearchCV(SVC(), param_grid = param_grid, cv=5, scoring = acc_scorer)

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Training Accuracy ', model.best_score_)

v5 = accuracy_score(y_test, pred)

v6 = f1_score(y_test, pred)

print('Best Parameter', model.best_params_)

print('Testing Accuracy Score', v5)

print('Testing F1 Score', v6)

accuracy_scoring['SVC Testing Accuracy Score'] = v5

f1_scoring['SVC Testing f1 Score'] = v6
print('Gaussian NB')

param_grid = {}



model = GridSearchCV(GaussianNB(), param_grid = param_grid, cv=5, scoring = acc_scorer)

model.fit(x_train, y_train)

pred = model.predict(x_test)

print('Training Accuracy ', model.best_score_)

v7 = accuracy_score(y_test, pred)

v8 = f1_score(y_test, pred)



print('Testing Accuracy Score', v7)

print('Testing F1 Score', v8)

accuracy_scoring['GaussianNB Testing Accuracy Score'] = v7

f1_scoring['GaussianNB Testing f1 Score'] = v8
for i, j in enumerate(accuracy_scoring):

    print(j, accuracy_scoring[j])
for i, j in enumerate(f1_scoring):

    print(j, round(f1_scoring[j], 2))
model = SVC(C= 10, kernel= 'linear')

model.fit(x_train, y_train)

pred = model.predict(x_test)



print('Training Accuracy Score ', accuracy_score(y_test, pred))

print('Testing F1 Score ', round(f1_score(y_test, pred), 2))

print('Testing Precision Score ', round(precision_score(y_test, pred), 2))