# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df.head()

df['Name'] = df.Name.map(lambda x: x.split(",")[1].split(".")[0])
df.head()
df.dtypes
df[['Sex', 'Embarked', 'Name', 'Ticket','Cabin']].describe()
def bar_chart(feature):

    survived = df[df['Survived']==1][feature].value_counts()

    dead = df[df['Survived'] == 0][feature].value_counts()

    df_res = pd.DataFrame([survived, dead])

    df_res.index = ['Survived','Dead']

    df_res.plot(kind='bar', stacked=True, figsize=(10,5))
bar_chart('Sex')
X = df[[ 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Fare', 'Parch', 'Embarked','Cabin','Ticket']]

y = df['Survived']
X.isna().any()
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass', y='Age', data=X, palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
X['Age'] = X[['Age','Pclass']].apply(impute_age, axis=1)
sns.heatmap(X.isnull(),yticklabels=False, cbar=False, cmap='viridis')
X.Cabin.isnull().sum() #204 not Nulls

X.Cabin.fillna(X.Cabin.mode()[0], inplace=True)

X.Embarked.fillna(X.Embarked.mode()[0], inplace=True)
X.isna().any()
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import category_encoders as ce
print(X.Name.unique())

X.Name = LabelEncoder().fit_transform(X.Name)

print(X.head())
embark = pd.get_dummies(X['Embarked'], drop_first=True)

sex = pd.get_dummies(X['Sex'], drop_first=True)
X.head()
X.drop(['Sex', 'Embarked'],axis=1, inplace=True)

X.head()
X = pd.concat([X, sex, embark], axis=1)
X.head()
X.Cabin = LabelEncoder().fit_transform(X.Cabin)

print(X.head())
X.Ticket = LabelEncoder().fit_transform(X.Ticket)

print(X.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=False)

print(X_train.head())
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 

model = LogisticRegression()

model.fit(X_train, y_train)

model.predict(X)
model.score(X,y)