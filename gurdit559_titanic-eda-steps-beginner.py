# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv('../input/titanic/train.csv',na_values="NAN")

test=pd.read_csv('../input/titanic/test.csv',na_values="NAN")

train.head()
test.head()
train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')
train.shape
test.shape
train.columns
train.info()
survived=train[train["Survived"]==1]

survived.head()
train['Sex'].value_counts()
female_s=survived[survived["Sex"]=="female"]

female_s.head()
female_s.shape
male_s=survived[survived['Sex']=="male"]

male_s.head()
male_s.shape
print("Total number of female who did not survive the crash is 81 out of 314 ")

print("Total number of male who did not survive the crash is 468 out of 577")
survived.shape
sns.set_style('whitegrid')

sns.countplot(x="Survived",data=train)
sns.set_style("whitegrid")

sns.countplot(x="Survived",hue=train.Sex,data=train)
sns.countplot(x="Pclass",hue=train.Sex,data=train)
sns.set_style("whitegrid")

sns.countplot(x="Survived",hue="Pclass",data=train)
sns.distplot(train["Age"].dropna(),kde=False,color="darkred",bins=40)
sns.distplot(survived["Age"].dropna(),kde=False,color="darkred",bins=40)
sns.countplot(x="SibSp",data=train)
sns.countplot(x="SibSp",data=survived)
train["Fare"].hist(bins=40,figsize=(8,4))
plt.figure(figsize=(12,6))

sns.boxplot(x="Pclass",y="Age",data=train)
def replace_age(cols):

    age=cols[0]

    pclass=cols[1]

    if pd.isnull(age):

        if pclass==1:

            return 37

        elif pclass==2:

            return 29

        elif pclass==3:

            return 24

    else:

        return age
train["Age"]=train[["Age","Pclass"]].apply(replace_age,axis=1)
train.isnull().sum()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="plasma")
train.drop("Cabin",axis=1,inplace=True)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="plasma")