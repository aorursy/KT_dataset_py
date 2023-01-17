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



train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")



train.head()
#We will start with the count of the number of survivors. 1 represents dead and 0 represents survived.

import seaborn as sb

import matplotlib.pyplot as plt

sb.countplot('Survived', data = train )

plt.title("Count of the survivors.")



#To check the survivors based on the sex of the passengers.

sb.catplot(x='Sex', col='Survived', kind='count', data=train)

plt.title("survivors based on the sex of the passengers.")
#Using SibSp to get the number of people who survived the incident



train[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar()

sb.countplot('SibSp',hue='Survived',data=train,)

plt.show()
#Let’s check the survival on Embarkment i.e. S,C and Q



sb.catplot(x='Survived', col='Embarked', kind='count', data=train)
#Class wise distribution of the data.

sb.barplot(x="Sex",y="Survived", hue="Pclass", data=train)

plt.title("Class wise distribution")
#Distribution of the data based on the age and sex distribution

g = sb.FacetGrid(train, col="Survived",row="Sex", hue = "Pclass")

g = g.map(plt.hist,"Age")
#The age distribution frequency.

plt.hist("Age", data = train)

plt.title("Age Distribution")