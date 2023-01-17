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
#importing required libraries

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns

sns.set()
#loading the files 

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test  = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.info()
test.head(), test.info()
train.isnull().sum(), test.isnull().sum()
train.describe(), test.describe()
def barPlot(feature):

    survived = train[train.Survived == 1][feature].value_counts()

    dead     = train[train.Survived == 0][feature].value_counts()

    df       = pd.DataFrame([survived, dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar', stacked = True, figsize=(10,5))
barPlot('Sex')
barPlot('Pclass')
barPlot('SibSp')
barPlot('Parch')
barPlot('Embarked')
train_test_combined = [train, test]



for dataset in train_test_combined:

    dataset["Title"] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
train.Title.value_counts()
test.Title.value_counts()
title_mapping = {'Mr': 0, 'Miss': 1,'Mrs': 2, 'Master': 3, 'Dr': 3,'Rev': 3, 'Col': 3, 'Dona': 3, 'Ms': 3,

                 'Major': 3,'Mlle': 2,'Mme': 1,'Countess'  :1,'Sir': 1,'Jonkheer': 1,'Ms': 1,'Don': 1,'Capt': 1,

                 'Lady': 1

                }



for dataset in train_test_combined:

    dataset['Title'] = dataset['Title'].map(title_mapping)
train.Title.value_counts(), test.Title.value_counts()
barPlot('Title')
#Since we derived Title from Name, we can remove the Name feature from the Datasets

train.drop(["Name"], axis = 1, inplace = True)

test.drop(["Name"], axis = 1, inplace = True)
sex_mapping = {'male': 0, 'female': 1}

for dataset in train_test_combined:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
#there are Null values in Age column. impute Age values

# using median values for age based on Title to impute 



train['Age'].fillna(train.groupby('Title')['Age'].transform("median"), inplace = True)

test['Age'].fillna(test.groupby('Title')['Age'].transform("median"), inplace = True)
for dataset in train_test_combined:

    dataset.loc[ (dataset["Age"] <= 16), 'Age' ] = 0

    dataset.loc[ (dataset["Age"] > 16) & (dataset["Age"] <= 26), 'Age' ] = 1

    dataset.loc[ (dataset["Age"] > 26) & (dataset["Age"] <= 36), 'Age' ] = 2

    dataset.loc[ (dataset["Age"] > 36 ) & ( dataset["Age"] <= 62), 'Age' ] = 3

    dataset.loc[ (dataset["Age"] > 62), 'Age' ] = 4
#Embarked

PClass1 = train.loc[train['Pclass']==1, 'Embarked'].value_counts()

PClass2 = train.loc[train['Pclass']==2, 'Embarked'].value_counts()

PClass3 = train.loc[train['Pclass']==3, 'Embarked'].value_counts()

df      = pd.DataFrame([PClass1, PClass2, PClass3])

df.index= ['1st Class','2nd Class','3rd Class']

df.plot(kind='bar', stacked = True, figsize=(10,5))
#Fill out missing value for Embark as S as majority of data set is from S 

for dataset in train_test_combined:

    dataset['Embarked'].fillna('S', inplace = True)

    

embark_mapping = {"S":0, "C":1, "Q":2}

for dataset in train_test_combined:

    dataset["Embarked"] = dataset["Embarked"].map(embark_mapping)
# Fare

train["Fare"].fillna(train.groupby('Pclass')['Fare'].transform("median"), inplace = True)

test["Fare"].fillna(test.groupby('Pclass')['Fare'].transform("median"), inplace = True)
for dataset in train_test_combined:

    dataset.loc[ dataset["Fare"]<=17, "Fare"] = 0

    dataset.loc[ (dataset["Fare"]>17) & (dataset["Fare"]<=30), "Fare"] = 1

    dataset.loc[ (dataset["Fare"]>30) & (dataset["Fare"]<=100), "Fare"] = 2

    dataset.loc[ dataset["Fare"]> 100, "Fare"] = 3
#Cabin

for dataset in train_test_combined:

    dataset["Cabin"] = dataset["Cabin"].str[:1]
cabin_mapping = {'A': 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2.0, "G": 2.4, "T": 2.8}

for dataset in train_test_combined:

    dataset["Cabin"] = dataset["Cabin"].map(cabin_mapping)

    

train.Cabin.unique()
train.Cabin.fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace = True)

test.Cabin.fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace = True)
#FamilySize

train["FamilySize"] = train.Parch + train.SibSp + 1

test["FamilySize"] = test.Parch + test.SibSp + 1
family_mapping = {1:0.0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}

for dataset in train_test_combined:

    dataset["FamilySize"] = dataset["FamilySize"].map(family_mapping)
drop_features = ['Parch','SibSp', 'Ticket']

for dataset in train_test_combined:

    dataset.drop(drop_features, axis = 1, inplace = True)
#Train Data

train_data = train.drop(["PassengerId","Survived"], axis = 1)

target = train["Survived"]

test_data = test.drop(["PassengerId"], axis = 1)
#FINAL FEATURES

train_data.head(4), test_data.head(4), train_data.shape, test_data.shape
from sklearn.linear_model import LogisticRegression

lm1 = LogisticRegression()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data, target, test_size = 0.3, random_state = 42)
lm1.fit(X_train, y_train)

y_pred = lm1.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

confusionmatrix =  confusion_matrix(y_test, y_pred)

confusionmatrix
def printConfusionMatrix(cm):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(cm)

    ax.grid(False)

    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))

    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))

    ax.set_ylim(1.5, -0.5)

    for i in range(2):

        for j in range(2):

            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')

    plt.show()
printConfusionMatrix(confusionmatrix)
print(classification_report(y_test, lm1.predict(X_test)))
lm1.fit(train_data, target)

test_target = lm1.predict(test_data)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_target})

submission.to_csv('submission.csv', index = False, )
df = pd.read_csv('submission.csv')

df.head(-1)