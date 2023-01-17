# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
data.head()
len(data)
data.Country.value_counts()
data.Sex.value_counts()
plt.figure(figsize=(12, 6))

data.groupby('Age')['PassengerId'].count().plot()
sns.countplot(data.Category)
# C = Crew, P = Passenger
sns.countplot(data.Survived)
data.isnull().sum()
# No null values
sns.catplot(x="Sex", y="Survived", kind="bar", data=data)
sns.catplot(x="Category", y="Survived", kind="bar", data=data)
grid = sns.FacetGrid(data, row='Survived', size=3, aspect=1.6)

grid.map(sns.distplot, 'Age', 'Survived')

grid.add_legend()
# Most young people survived
data.groupby('Survived')['Age'].mean()
data.groupby('Country')['Survived'].mean().reset_index().sort_values('Survived', ascending=False)
# Convert categorical features
data['Sex'] = data.Sex.astype('category').cat.codes

data['Category'] = data.Category.astype('category').cat.codes

data['Country'] = data.Country.astype('category').cat.codes
sns.distplot(data.Age, bins=8)
data['AgeBin'] = pd.cut(data['Age'], 10)

data[['AgeBin', 'Survived']].groupby(['AgeBin'], as_index=False).mean().sort_values(by='AgeBin', ascending=True)
# ~50% of surviving between 17 and 35
data.loc[ data['Age'] <= 10, 'Age'] = 0

data.loc[(data['Age'] > 10) & (data['Age'] <= 20), 'Age'] = 1

data.loc[(data['Age'] > 20) & (data['Age'] <= 30), 'Age'] = 2

data.loc[(data['Age'] > 30) & (data['Age'] <= 40), 'Age'] = 3

data.loc[(data['Age'] > 40) & (data['Age'] <= 50), 'Age'] = 4

data.loc[(data['Age'] > 50) & (data['Age'] <= 60), 'Age'] = 5

data.loc[(data['Age'] > 60) & (data['Age'] <= 70), 'Age'] = 6

data.loc[ data['Age'] > 70, 'Age'] = 7
# Remove PassengerId, Firstname and Lastname

data = data[['Country', 'Sex', 'Age', 'Category', 'Survived']]
data.head()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
X = data.drop("Survived", axis=1)

Y = data["Survived"]
# Split 20% test, 80% train



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
# Logistic Regression



log = LogisticRegression(max_iter=100)

log.fit(X_train, Y_train)

Y_pred_log = log.predict(X_test)

acc_log = accuracy_score(Y_pred_log, Y_test)

acc_log
rf = RandomForestClassifier()



# search the best params

grid = {'n_estimators':[100,200], 'max_depth': [2,5,10]}



clf_rf = GridSearchCV(rf, grid, cv=10)

clf_rf.fit(X_train, Y_train)



Y_pred_rf = clf_rf.predict(X_test)

# get the accuracy score

acc_rf = accuracy_score(Y_pred_rf, Y_test)

print(acc_rf)
# The best model is the Random Forest