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
import pandas as pd

import nltk

import re

%matplotlib inline

import random

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

from IPython.display import display

from sklearn import datasets

from sklearn.svm import SVC

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

full_data = [df_train,df_test]

for dataset in full_data:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(df_train['Title'], df_train['Sex'])
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
df_train['Age'] = df_train['Age'].fillna(-1)

df_test['Age'] = df_test['Age'].fillna(-1)  

full_data = [df_train,df_test]



for dataset in full_data:

    

    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Master'), 'Age'] = 4.57

    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Miss'), 'Age'] = 21.84

    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Mr'), 'Age'] = 32.36

    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Mrs'), 'Age'] = 35.78

    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Rare'), 'Age'] = 45.54

    dataset['Age'] = dataset['Age'].astype(int)  

    

    full_data = [df_train, df_test]

for dataset in full_data:

    

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 7

df_train[['Sex','Age','Survived']].groupby(['Sex','Age'],as_index=False).mean()
df_train['Family_members'] = df_train['SibSp'] + df_train['Parch']

df_test['Family_members'] = df_test['SibSp'] + df_test['Parch']

df_train[['Family_members','Survived']].groupby(['Family_members'],as_index=False).mean()
for data in full_data: #nan 값을 U로 다 채우겠습니다.

    data['Cabin'].fillna('U', inplace=True)

    data['Cabin'] = data['Cabin'].apply(lambda x: x[0])

    data['Cabin'].unique()

    data['Fare'].fillna(0,inplace=True)

    data['Fare'] = data['Fare'].apply(lambda x: int(x))
df_train['Cabin']

temp = df_train['Fare'].unique()

temp.sort()

temp
for dataset in full_data:

    dataset.loc[ dataset['Fare']<=30, 'Fare'] = 0

    dataset.loc[(dataset['Fare']>30)&(dataset['Fare']<=80), 'Fare'] = 1

    dataset.loc[(dataset['Fare']>80)&(dataset['Fare']<=100), 'Fare'] = 2

    dataset.loc[(dataset['Fare']>100), 'Fare'] = 3
# 임의의 U값에 가격에 맞게끔 'Cabin'값을 측정한다.

for dataset in full_data:

    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 0), 'Cabin'] = 'G'

    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 1), 'Cabin'] = 'T'

    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 2), 'Cabin'] = 'C'

    dataset.loc[(dataset['Cabin'] == 'U')&(dataset['Fare'] == 3), 'Cabin'] = 'B'



df_train[['Cabin','Survived']].groupby(['Cabin'],as_index=False).mean()
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S') #'S'로 다 채워놓자.
df_train.info()

df_test.info()
features = ["Pclass", "Sex", "Family_members", "Age", "Cabin", "Embarked", "Fare"]

X = pd.get_dummies(df_train[features])

X_test = pd.get_dummies(df_test[features])

y = df_train["Survived"]
svm=SVC(C=1, kernel='rbf', coef0=1)

svm

svm.fit(X, y)
svm.score(X, y)

predictions = svm.predict(X_test)

output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission01.csv', index=False)

print("Your submission was successfully saved!")