# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read the train data

df_train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train_data.head()

df_train_data.shape

Survived = df_train_data['Survived']

Survived
#Read the test data

df_test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test_data.shape

#pd.set_option('display.max_rows', None)

#df_test_data
#Need to drop survived column from the training dataset

df_train_data = df_train_data.drop('Survived', axis = 1)

df_train_data.head()
#Concatenate both dataframes together to perform feature engineering all at once

df_all = pd.concat([df_test_data, df_train_data], ignore_index = True)

df_all
#See how many missing values there are per column

df_all.isnull().sum(axis = 0)
#Want to see which variables have the highest correlation between each other in order to make more infromed decisions about our missing value imputation

pd.DataFrame.corr(df_all, method ='pearson')
#Dropping columns with greater than 50% missing vlaues

#df_all = df_all.dropna(thresh=0.7, axis = 1)

#Dropping rows with greater than 50% missing values

#df_all = df_all.dropna(thresh=0.7)

df_all = df_all.drop(['Cabin'], axis=1)

df_all.shape
#Fill NA 

#categorical variables

df_all['Embarked'] = df_all['Embarked'].fillna(df_all['Embarked'].mode().loc[0])



#continuous varialbles

df_all['Fare'] = df_all['Fare'].fillna(df_all['Fare'].median())

df_all['Age'] = df_all['Age'].fillna(df_all['Age'].median())



df_all.isnull().sum(axis = 0)
## Assign Binary to Sex str

df_all['Sex'] = df_all['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Title

#df['Title'] = df['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Rare':4} ).astype(int)

# Embarked

df_all['Embarked'] = df_all['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)



# Get Rid of Ticket variable

df_all= df_all.drop(['Ticket'], axis=1)
df_all.dtypes
#Splin data to create models

test_df = df_all.loc[0:417]

train_df = df_all.loc[418:1308]

train_df = train_df.reset_index()

train_df['Survived'] = Survived

train_df
train_df['Survived']
y = train_df['Survived']
from sklearn.ensemble import RandomForestClassifier



#y = train_df[Survived]



features = (list(test_df.columns))

features.remove('Name')

X = (train_df[features])

X_test = (test_df[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

y_predict = model.predict(X_test)

model.score(X, y)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_predict})

output.to_csv('my_submission.csv', index=False)

#print("Your submission was successfully saved!")
model.score(X, y)