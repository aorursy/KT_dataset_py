# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import matplotlib.pyplot as plt

import seaborn as sns  





from numpy import reshape

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# %% read csv

titanic = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

y_test = pd.read_csv("../input/gender_submission.csv")
#%% Female-Male range



titanic.groupby('Sex')['Survived'].value_counts().plot(kind='bar', stacked=True, colormap='winter')

plt.show()

#%% Sex- Survived Plot



sex_survived = titanic.groupby(['Sex', 'Survived'])

sex_survived.size().unstack().plot(kind='bar', stacked=True, colormap='winter')

plt.ylabel('Frequency')

plt.title('Survivings by sex')

plt.show()

#%% Kid Female-Male



male=len(titanic.groupby('Sex').groups['male'])



female=len(titanic.groupby('Sex').groups['female'])



kid_passenger = titanic[titanic['Age'] < 16]



kid_survived = kid_passenger.groupby(['Sex', 'Survived'])

kid_survived.size().unstack().plot(kind='bar', stacked=True, colormap='winter')

plt.ylabel('Frequency')

plt.title('Kids Survived - sex')

plt.show()
#%% Age - Survived  Plot

class_survived = titanic.groupby(['Age', 'Survived'])

class_survived.size().unstack().plot(kind='bar', stacked=True, colormap='winter')

plt.xlabel('Ages')

plt.ylabel('Frequency')

plt.title('Survived by passenger class')

plt.show()
#%% Fare levels 

titanic['FareLevel'] = pd.qcut(titanic['Fare'], 10)

titanic['FareLevel'].value_counts().sort_values(ascending= False)



titanic[['FareLevel', 'Survived']].groupby(['FareLevel'],as_index=False).mean().sort_values(by='FareLevel',ascending=True)

#%% Fare- Survived Plot



fare_survived = titanic.groupby(['FareLevel', 'Survived'])

fare_survived.size().unstack().plot(kind='bar', stacked=True, colormap='winter')

plt.ylabel('Frequency')

plt.title('Survivings by fare')

plt.show()
#%% Embarked

titanic['Embarked'].value_counts().plot(kind='bar')

plt.title('Embarking ports')

plt.ylabel('Frequency')

plt.xlabel('S=Southampton, C=Cherbourg, Q=Queenstown')

plt.show()
#%% Embarked - Survived  Plot

class_survived = titanic.groupby(['Embarked', 'Survived'])

class_survived.size().unstack().plot(kind='bar', stacked=True, colormap='winter')

plt.xlabel('1 = Southampton,   2 = Cherbourg,   3 = Queenstown')

plt.ylabel('Frequency')

plt.title('Survived by Embarked')

plt.show()
#%% PassengerClass

titanic['Pclass'].value_counts().plot(kind='bar')

plt.title('PassengerClass')

plt.ylabel('Frequency')

plt.xlabel('S=Southampton, C=Cherbourg, Q=Queenstown')

plt.show()

#%% PassengerClass - Survived  Plot

class_survived = titanic.groupby(['Pclass', 'Survived'])

class_survived.size().unstack().plot(kind='bar', stacked=True, colormap='winter')

plt.xlabel('1st = Upper,   2nd = Middle,   3rd = Lower')

plt.ylabel('Frequency')

plt.title('Survived by passenger class')

plt.show()
#%% adding with family or not



titanic['family'] = titanic['SibSp'] + titanic['Parch']



titanic = titanic.append(titanic.iloc[:0])



titanic.family = [1 if each > 0 else 0 for each in titanic.family]





test['family'] = test['SibSp'] + test['Parch']



test = test.append(test.iloc[:0])



test.family = [1 if each > 0 else 0 for each in test.family]
#%% Dropping





y_test.drop(["PassengerId"],axis=1,inplace = True)





embarked_counts = titanic.Embarked.value_counts(normalize = True)



sex_counts = titanic.Sex.value_counts(normalize = True)





titanic.Sex = [1 if each == "female" else 0 for each in titanic.Sex]

titanic.drop(["Name","Ticket","Cabin","SibSp","Parch","Fare","FareLevel"],axis=1,inplace = True)

titanic.Embarked = [1 if each == "S" else each for each in titanic.Embarked]

titanic.Embarked = [2 if each == "C" else each for each in titanic.Embarked]

titanic.Embarked = [3 if each == "Q" else each for each in titanic.Embarked]

titanic.Embarked = [each if each == "Q" or "C" or "S" else 0 for each in titanic.Embarked]





test.Sex = [1 if each == "female" else 0 for each in test.Sex]

test.drop(["Name","Ticket","Cabin","SibSp","Parch","Fare"],axis=1,inplace = True) 

test.Embarked = [1 if each == "S" else each for each in test.Embarked]

test.Embarked = [2 if each == "C" else each for each in test.Embarked]

test.Embarked = [3 if each == "Q" else each for each in test.Embarked]

test.Embarked = [each if each == "Q" or "C" or "S" else 0 for each in test.Embarked]





titanic = titanic.fillna(value=0)



x_test = test.fillna(value=0)





print(titanic.info())



y_train = titanic.Survived.values



x_train = titanic.drop(["Survived"],axis=1)



y_train = y_train.reshape(-1,1)





stat=titanic.describe()



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
#%% Score

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

print("test accuracy {}".format(lr.score(x_test,y_test)))