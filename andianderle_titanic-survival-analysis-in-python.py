# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.info()

print("----------------------------")

test.info()
train["Title"]=(train["Name"].str.extract(r'([A-Z][a-z]+)\.'))
train.set_index("PassengerId")
train[train["Cabin"].isnull()]
# survival rate depending on embarkment

sns.factorplot('Embarked','Survived', data=train,size=3.65,aspect=3)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(13,4))



sns.countplot(x='Embarked', data=train, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)



# survival rate depending on title

sns.factorplot('Title','Survived', data=train,size=3.65,aspect=3)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(13,4))



sns.countplot(x='Title', data=train, ax=axis1)

sns.countplot(x='Survived', hue="Title", data=train, order=[1,0], ax=axis2)
cmap=cm.swirly_cmap

plt.matshow(train.corr())

#plt.colorbar()
train.groupby(["Title"]).size()
train[train["Survived"]==1].groupby(["Title"]).size()
train["Title"].value_counts()
train.groupby('Title')['Survived'].sum()