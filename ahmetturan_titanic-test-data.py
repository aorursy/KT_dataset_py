# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
g_sub=pd.read_csv("../input/gender_submission.csv")

train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
train.info()
train.columns
train.head(10)
train.tail(10)
train.corr()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(train.corr(),annot=True,ax=ax,linewidths=5,fmt=".3f")
train.plot(y="Age",kind="hist",grid=True,title="AGE GRAPHİC",alpha=1,color="red",bins=40,figsize=(15,15))
train.isnull().sum()
sns.countplot(x="Pclass",data=train,hue="Survived",color="red")
sns.countplot(x="Embarked",hue="Pclass",data=train)
sns.countplot(x="Sex",data=train,hue="Survived")
train["Sex"].value_counts(normalize=True)
train[train["Fare"]==train["Fare"].max()]
f,ax=plt.subplots(figsize=(18,18))

sns.countplot(x="Age",data=train,hue="Survived")
f,ax=plt.subplots(figsize=(20,20))

sns.countplot(x="Age",data=train,hue="Pclass")
test