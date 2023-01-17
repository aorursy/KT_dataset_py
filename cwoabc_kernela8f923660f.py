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
train = pd.read_csv('../input/titanic/train.csv')
train.describe
train.corr()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x = 'SibSp', hue = "Survived", data =train)

plt.legend(loc = "upper right", title ="Survived ~ SibSp")
sns.distplot(train[train['Survived'] == 0].Fare, kde =False,rug =False)

## <matplotlib.axes._subplots.AxesSubplot object at 0x7fd3b9839160>

sns.distplot(train[train['Survived'] == 1].Fare, kde =False,rug =False)

## <matplotlib.axes._subplots.AxesSubplot object at 0x7fd3b9839160>
train.isnull().sum()
train.drop(['PassengerId','Name','Cabin','Ticket',], axis=1, inplace=True)

train["Age"].fillna(train["Age"].median(skipna=True), inplace=True)

train["Embarked"].fillna(train['Embarked'].value_counts().idxmax(), inplace=True)