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
#Loading Dataset

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.info()
train.isnull().sum()
train.shape
train.ndim
train.dtypes
train.describe()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18,6))



plt.subplot2grid((2,3),(0,0))

train.Survived.value_counts(normalize=True).plot(kind="bar",alpha=0.6)

plt.title("Survived")



plt.subplot2grid((2,3),(0,1))

plt.scatter(train.Survived, train.Age, alpha=0.1)

plt.title("Age wrt Survived")



plt.subplot2grid((2,3),(0,2))

train.Pclass.value_counts(normalize=True).plot(kind="bar",alpha=0.6)

#plt.scatter(train.Pclass, train.Age, alpha=0.1)

plt.title("Class")



plt.subplot2grid((2,3),(1,0), colspan=2)

for x in [1,2,3]:

    train.Age[train.Pclass == x].plot(kind="kde")

plt.title("class wrt Age")

plt.legend(("1st","2nd","3rd"))



plt.subplot2grid((2,3),(1,2))

train.Embarked.value_counts(normalize=True).plot(kind="bar",alpha=0.5)

plt.title("Embarked")



plt.show()