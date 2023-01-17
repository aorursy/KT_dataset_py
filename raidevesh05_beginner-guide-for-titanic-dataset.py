#importing required libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading the training and test data

titanic_train = pd.read_csv("../input/train.csv")

titanic_test = pd.read_csv("../input/test.csv")



# The first thing after reading the dataset is to know the dimensions.

print("The dimensions of training data and test data is",titanic_train.shape,"and",titanic_test.shape,"respectively")
titanic_train.info()





sns.set_style('whitegrid')



sns.countplot(x='Survived', data= titanic_train)



#people who survived v/s who didn't
sns.countplot(x='Survived', hue='Sex', data= titanic_train,palette='RdBu_r')
sns.countplot(x='Survived', hue='Pclass', data= titanic_train, palette='rainbow')



def impute_cabin(col):

   Cabin = col[0]

   if type(Cabin) == str:

       return 1

   else:

       return 0



titanic_train['Cabin'] = titanic_train[['Cabin']].apply(impute_cabin, axis = 1)
titanic_train['Cabin'].describe()
age_avg=titanic_train['Age'].mean()

age_std=titanic_train['Age'].std()



import random

random_list = np.random.randint(age_avg - age_std, age_avg + age_std )

titanic_train['Age'][np.isnan(titanic_train['Age'])] = random_list

titanic_train['Age'] = titanic_train['Age'].astype(int)
titanic_train['Age'].describe()
titanic_train["Embarked"]=titanic_train["Embarked"].fillna("S")
titanic_train['family_size'] = titanic_train['SibSp'] + titanic_train['Parch'] + 1

titanic_train['is_alone'] = 0

titanic_train.loc[titanic_train['family_size'] == 1, 'is_alone'] = 1

train=titanic_train.copy()

     #Mapping Sex

sex_map = { 'female':0 , 'male':1 }

train['Sex'] = train['Sex'].map(sex_map).astype(int)



    #Mapping Embarked

embark_map = {'S':0, 'C':1, 'Q':2}

train['Embarked'] = train['Embarked'].map(embark_map).astype(int)
train.head(5)
train=train.drop("Name",axis=1)

train=train.drop("Ticket",axis=1)
train.describe(include='all')
Y=train.Survived

train=train.drop("Survived",axis=1)


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train, Y, test_size=0.33, random_state=42)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss

from xgboost import XGBClassifier

classifier =  XGBClassifier(n_estimators=1000, learning_rate=0.05,n_jobs=-1)

classifier.fit(X_train, y_train)

pred3 = classifier.predict(X_valid)



print(classification_report(y_valid, pred3))

print('\n')

print(confusion_matrix(y_valid, pred3))

print('\n')

print(accuracy_score(y_valid, pred3))