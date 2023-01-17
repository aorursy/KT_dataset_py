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
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.linear_model import LogisticRegression
train_df.head()
train_df.describe()
train_df.info()
test_df.describe()
test_df.info()
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.countplot(train_df['Survived'], palette='RdBu_r')
survived = train_df[train_df['Survived']==1]['Survived'].sum()

survived

sns.countplot(train_df['Survived'], hue=train_df['Sex'], palette='rainbow')
sns.countplot(train_df['Survived'], hue=train_df['Pclass'], palette='rainbow')

train_df['Age'].hist(color='darkred', bins=30, alpha=0.6)
train_df['Fare'].hist(color='purple', bins=30, figsize=(8,4))

plt.figure(figsize=(12,7))

sns.boxplot(x=train_df['Pclass'], y=train_df['Age'], palette='winter')
sns.boxplot(x=train_df['Pclass'], y=train_df['Fare'])
train_df.isnull().sum()
train_df['Age'] = train_df['Age'].fillna(train_df.groupby('Pclass')['Age'].transform('mean'))
train_df.drop('Cabin', axis=1, inplace=True)

train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df.isnull().sum()
test_df.isnull().sum()
test_df['Age'] = test_df['Age'].fillna(train_df.groupby('Pclass')['Age'].transform('mean'))
test_df.drop('Cabin',axis=1, inplace=True)
test_df['Fare'] = test_df['Fare'].fillna(train_df.groupby('Pclass')['Fare'].transform('mean'))
test_df.isnull().sum()
sex = pd.get_dummies(train_df['Sex'], drop_first=True)

embark = pd.get_dummies(train_df['Embarked'], drop_first=True)
train_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis =1, inplace=True)
train_df = pd.concat([train_df, sex, embark], axis=1)
train_df.head(2)
sex = pd.get_dummies(test_df['Sex'], drop_first=True)

embark = pd.get_dummies(test_df['Embarked'], drop_first=True)
test_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis= 1, inplace=True)
test_df.head(2)
test_df = pd.concat([test_df, sex, embark], axis =1)
test_df.head(2)
X_train = train_df.drop(['Survived','PassengerId'], axis=1)

y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1)



X_train.shape, y_train.shape, X_test.shape
logmodel = LogisticRegression(max_iter=200)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': predictions

})



submission.to_csv('submission.csv', index = False)

submission.head()
