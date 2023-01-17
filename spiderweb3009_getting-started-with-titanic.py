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
import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("../input/titanic/train.csv")

df.head()
df.describe()
df.isnull().sum()
df['Age'].fillna(df['Age'].mean(), inplace=True)

df.isnull().sum()
df.corr()
ax = plt.figure(figsize=(12,12))

#sns.scatterplot(df['Cabin'], df['Fare'], hue= df['Survived'])

sns.barplot(df['Cabin'], df['Fare'], hue= df['Survived'])
df = df.drop(columns=["Cabin", "Embarked", "Name", "Ticket"])
sns.countplot(df['Sex'], hue=df['Survived'])
f, axes = plt.subplots(1,2, figsize=(12,6))

sns.barplot(df['Pclass'], df['Fare'], hue=df['Survived'], ax=axes[0])

sns.scatterplot(df['Pclass'], df['Fare'], hue=df['Survived'], ax=axes[1])
sns.heatmap(df.corr(), annot=True)
from sklearn.preprocessing import LabelEncoder



df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
x = df.drop(columns=["PassengerId", "Survived"])

y = df['Survived']
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(random_state=3)

model.fit(x, y)
df_test = pd.read_csv("../input/titanic/test.csv")

df_test = df_test.drop(columns=["Cabin", "Embarked", "Name", "Ticket"])

df_test.head()
df_test['Sex'] = LabelEncoder().fit_transform(df_test['Sex'])

df_test.isnull().sum()
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)

df_test.isnull().sum()
pred = model.predict(df_test.drop(columns=['PassengerId']))
sns.countplot(df_test['Pclass'], hue=pred)
output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': pred})

output.to_csv('my_submission.csv', index=False)