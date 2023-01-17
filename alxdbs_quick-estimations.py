#The goal of this workbook is to work on Titanic data set to predict if one survived or not.

#Methodology is :

    #1. Import python libraries for ML

    #2. Load Data from Data set provided in Kaggle

    #3. Train a classification algorithm

    #4. Give predictions
#Part 1: Import python Libraries

import pandas as pd

import numpy as np

import random

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
#Part 2: Load data set

raw_titanic_data =  pd.read_csv('../input/train.csv')
raw_titanic_data.head()
raw_titanic_data.count()
#Data are shuffled to avoid any selection in the raw data

sh_titanic_data = shuffle(raw_titanic_data)
sh_titanic_data.head()
#Data set has been loaded but it need to be separated in 2 parts: 

    #- trainning set ~70%

    #- Cross validation ~30%

#Combination of this two data set will enable us to know if we are dealing with a bias or variance issue 

#Sometimes the data set is divided in 3 parts including the test set but here testy set is given in another csv file.  
nb_trainning_ex = int(sh_titanic_data['PassengerId'].count()*0.7)

nb_cv_ex = sh_titanic_data['PassengerId'].count() - nb_trainning_ex

print(nb_trainning_ex, nb_cv_ex, (nb_cv_ex + nb_trainning_ex)==sh_titanic_data['PassengerId'].count())
train_set = sh_titanic_data[1:nb_trainning_ex+1]

cv_set = sh_titanic_data[nb_trainning_ex:]
X_train = train_set.loc[:,['SibSp', 'Parch', 'Fare','Pclass']]

y_train = train_set['Survived']
X_cv = cv_set.loc[:,['SibSp', 'Parch', 'Fare','Pclass']]

y_cv = cv_set['Survived'] 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

clf = lr.fit(X_train,y_train)
clf.score(X_cv,y_cv)
print(lr.coef_)
#plt.plot(lr.coef_[0])

#plt.show()
#First conclusion : the score here is still low: this is a quick and dirty model

#To gain in prediction it is interesting to focus on other features which could impact our model
#Let's study 'Sex' feature: it seems that 'Sex' is always filled so we need only to convert 'Sex' to a boolean

train_set['Sex'].count(), train_set['Sex'].dtype
X_train['Sex'] = train_set['Sex'] == 'male'

X_cv['Sex'] = cv_set['Sex'] == 'male'
clf2 = lr.fit(X_train,y_train)

clf2.score(X_cv,y_cv)
#Adding Sex has improved our prediction, let's try to add 'Age'

#'Age' is not always filled up so we will add the median value when age is not filled
X_train['Age'] = train_set['Age'].fillna(train_set.Age.median())

X_cv['Age'] = cv_set['Age'].fillna(cv_set.Age.median())
clf3 = lr.fit(X_train, y_train)

clf3.score(X_cv, y_cv)
clf3.coef_
#Trying to have better precision

from sklearn.model_selection import cross_val_score

def compute_score(clf, X, y):

    xval = cross_val_score(clf, X, y, cv = 5)

    return mean(val)
lr.score(lr, X_train, y_train)