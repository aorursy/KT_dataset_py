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
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
gender_submission.head()
test.head()
train.head()
gender_submission.shape
test.shape
train.shape
test.dtypes
train.dtypes
gender_submission.describe()
test.describe()
train.describe()
#c=gender_submission.dtypes[gender_submission.dtypes=='object'].index
#gender_submission[c].describe
gender_submission.hist()
test.hist()
train.hist()
sns.pairplot(gender_submission,hue='Survived')
#sns.pairplot(train,hue='SibSp')
#sns.pairplot(test,hue='SibSp')
#sns.catplot(x='PassengerId',y='Survived',data=gender_submission)
gender_submission.plot(kind='density',subplots=True ,layout=(3,3),sharex=False)
test.plot(kind='density',subplots=True ,layout=(3,3), sharex=False)
train.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
gender_submission.plot(kind='box', subplots=True, layout=(5,5),sharex=False,sharey=False )
test.plot(kind='box',subplots=True,layout=(3,3),sharex=False,sharey=False)
train.plot(kind='box',subplots=False,layout=(3,3),sharex=True,sharey=True)
gender_submission['Survived'].value_counts().plot(kind='bar')
#test['Fare'].value_counts().plot(kind='bar')
#test['Age'].value_counts().plot(kind='bar')
test['Pclass'].value_counts().plot(kind='bar')
test['Sex'].value_counts().plot(kind='bar')
test['Embarked'].value_counts().plot(kind='bar')
train['Survived'].value_counts().plot(kind='bar')
train['SibSp'].value_counts().plot(kind='bar')
train['Pclass'].value_counts().plot(kind='bar')
train['Embarked'].value_counts().plot(kind='bar')
#train['Age'].value_counts().plot(kind='bar')
corr=gender_submission.corr()
sns.heatmap(corr,vmax=1,square=False)

corr=test.corr()

sns.heatmap(corr,vmax=1,square=False)
corr=train.corr()

sns.heatmap(corr,vmax=1,square=True)