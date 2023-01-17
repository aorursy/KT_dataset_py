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
%matplotlib inline

train=pandas.read_csv("../input/titanic/train.csv")
train.head(10)
test=pandas.read_csv("../input/titanic/test.csv")
test.head(10)
train.info()
train.describe()
test.info()
test.describe()
train['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
train['Sex'].value_counts().plot.pie(autopct='%1.2f%%')
train['Pclass'].value_counts().plot.pie(autopct='%1.2f%%')
plt.figure(figsize=(12,5))
plt.subplot(121)
train['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Num')

ply.subplot(122)
train.boxplot(colum='Age',showfliers=False)
plt.show()
train['Embarked'].value_counts().plot.pie(autopct='%1.2f%%')
train.groupby(['Sex','Survived'])['Sex'].count()
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
train.groupby(['Pclass','Survived'])['Pclass'].count()
train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
train[['Embarked','Survived']].groupby(['Embarked']).mean().plot.bar()
train["Age"]=train["Age"].fillna(train["Age"].median())
train["Embarked"]=train["Embarked"].fillna('S')
train.info()
test["Age"]=test["Age"].fillna(test["Age"].median())
test["Fare"]=test["Fare"].fillna(8.05)
test.describe()