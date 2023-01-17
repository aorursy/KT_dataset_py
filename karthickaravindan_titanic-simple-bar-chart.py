#package



import pandas as pd

import numpy as np

import os
df_train = pd.read_csv("../input/train.csv")

df_train_1 = df_train.copy()

df_train.head()
df_test = pd.read_csv("../input/test.csv")

df_test.head()
df_train.describe()
df_train.info()
df_train = df_train.drop(['Survived'],axis=1)

df_col = list(df_train.columns)

df_train_null = list(df_train.isnull().sum())

df_test_null = list(df_test.isnull().sum())
data={

        "Column name":df_col,

        "Train": df_train_null,

        "Test": df_test_null

     }



df_null = pd.DataFrame(data)

df_null
pclass = df_train["Pclass"].value_counts()

pclass
df_train["Age"].fillna(df_train.groupby("Pclass")["Age"].transform("median"), inplace=True)
df_train.isnull().sum()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

df_train = df_train_1
def bar_chart(feature):

    survived = df_train[df_train['Survived']==1][feature].value_counts()

    not_survived = df_train[df_train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,not_survived])

    df.index=['Survived','Not_survived']

    df.plot(kind='bar',stacked=True,fig=(18,6),title=feature)
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('Embarked')
bar_chart('SibSp')
bar_chart('Parch')
train=df_train.drop(['Survived'],1)

combine =  train.append(df_test) 

combine.reset_index(inplace=True)

combine.drop(['PassengerId','index'],1,inplace=True)

combine.head()
combine.shape
title = set()

for name in combine['Name']:

    title.add(name.split(",")[1].split(".")[0].strip())

print(title)