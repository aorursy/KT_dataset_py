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
#importing for data viz

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

import cufflinks as cf

cf.go_offline()

init_notebook_mode(connected=True)

%matplotlib inline

sns.set_style('whitegrid')
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
test_data.head()
train_data.info()
train_data.describe()
train_data.isnull()
plt.figure(figsize=(10,10))

sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis' ,yticklabels=False)
#Checking who survived

sns.countplot(x='Survived', data=train_data)
sns.countplot(x='Survived', hue='Sex', data=train_data)
sns.countplot(x='Survived', hue='Pclass', data=train_data)
plt.figure(figsize=(12,8))

sns.distplot(a=train_data['Age'].dropna(), bins=35)
sns.countplot(x='SibSp', data=train_data)
sns.countplot(x='Parch', data=train_data)
plt.figure(figsize=(10,8))

sns.distplot(a=train_data['Fare'], bins=35, kde=False)
train_data['Fare'].iplot(kind='hist', bins=50)
train_data.info()
#checking mean Age w.r.t. Pclass to fill-in missing values

train_data.groupby('Pclass')['Age'].mean()
def fillna(rec):

    Pclass = rec['Pclass']

    Age = rec['Age']

    if pd.isnull(Age):

        if Pclass == 1:

            return 38.23

        elif Pclass == 2:

            return 29.88

        else:

            return 25.14

    else:

        return Age
train_data['Age'] = train_data[['Pclass', 'Age']].apply(fillna, axis=1)
plt.figure(figsize=(10,10))

sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis' ,yticklabels=False)
train_data.drop('Cabin', axis=1, inplace=True)
#Dropping na (Few are in Embarked)

train_data.dropna(axis=0, inplace=True)
plt.figure(figsize=(10,10))

sns.heatmap(train_data.isnull(), cbar=False, cmap='viridis' ,yticklabels=False)
train_data = pd.get_dummies(train_data, columns=['Embarked', 'Sex'], drop_first=True)
train_data.drop(['Name', 'Ticket'], axis=1, inplace=True)
#Final Check

train_data
#Splitting predictor X and response y

X=train_data.drop('Survived', axis=1)

y=train_data['Survived']
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(solver='liblinear')
log_model.fit(X,y)
test_data
test_data.drop(['Name', 'Ticket'], axis=1, inplace=True)
#checking mean age for filling-in missing values

test_data.groupby('Pclass').mean()['Age']
def fillna_test(rec):

    Pclass = rec['Pclass']

    Age = rec['Age']

    if pd.isnull(Age):

        if Pclass == 1:

            return 41

        elif Pclass == 2:

            return 28.9

        else:

            return 24

    else:

        return Age
test_data['Age'] = test_data[['Pclass', 'Age']].apply(fillna_test, axis=1)
plt.figure(figsize=(10,10))

sns.heatmap(test_data.isnull(), cbar=False, cmap='viridis' ,yticklabels=False)
test_data.drop('Cabin', axis = 1, inplace=True)
test_data.dropna(inplace=True)
test_data.head()
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data.info()
predictions = log_model.predict(test_data)
predictions