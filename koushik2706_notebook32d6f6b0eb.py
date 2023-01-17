# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
data.head(5)
data.shape
data.isnull().sum()
data.nunique()
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
data.corr()
import seaborn as sns
import matplotlib.pyplot as plt
labels = ['Retained', 'Exited']
count = data['Exited'].value_counts()
plt.pie(count, labels = labels, autopct='%0.02f%%')
plt.show()
fig, axis = plt.subplots(3, 2, figsize=(10,10))
sns.countplot(x='Gender', hue='Exited', data=data, ax=axis[0,0])
sns.countplot(x='Geography', hue='Exited', data=data, ax=axis[0,1])
sns.countplot(x='Tenure', hue='Exited', data=data, ax=axis[1,0])
sns.countplot(x='NumOfProducts', hue='Exited', data=data, ax=axis[1,1])
sns.countplot(x='HasCrCard', hue='Exited', data=data, ax=axis[2,0])
sns.countplot(x='IsActiveMember', hue='Exited', data=data, ax=axis[2,1])
fig, axis = plt.subplots(2,2, figsize=(10,10))
sns.boxplot(x='Exited', y='Age', hue='Exited', data=data, ax=axis[0,0])
sns.boxplot(x='Exited', y='CreditScore', hue='Exited', data=data, ax=axis[0,1])
sns.boxplot(x='Exited', y='EstimatedSalary', hue='Exited', data=data, ax=axis[1,0])
sns.boxplot(x='Exited', y='Balance', hue='Exited', data=data, ax=axis[1,1])
dummies = pd.get_dummies(data, columns = ['Geography'])
dummies
def change(gender):
    if gender == 'Male':
        return 1
    else:
        return 0
dummies['Gender'] = dummies['Gender'].apply(change)
r = dummies.drop(['Exited'], axis = 1)
r['Exited'] = dummies['Exited']
r
x= dummies.iloc[:, 0:-1]
y = dummies.iloc[:, -1]
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(xtrain, ytrain)
from sklearn.linear_model import Perceptron
clf = Perceptron(penalty='l1', alpha=0.001, max_iter=1000)
clf.fit(xtrain, ytrain)
y_pred = clf.predict(xtest)
param_dist = {
    'penalty':['l1', 'l2', 'elasticnet'],
    'alpha':[0.1, 0.001, 0.0001],
    'max_iter':[1000, 10000, 100000]
}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(clf, param_grid = param_dist, cv = 10, n_jobs = -1)
grid.fit(xtrain, ytrain)
grid.best_estimator_
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ytest, y_pred)
accuracy

