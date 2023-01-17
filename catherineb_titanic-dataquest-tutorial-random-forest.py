# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load training set

titanic = pd.read_csv("../input/train.csv")



# import sklearn

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier



# clean data

# age

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

# sex

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# embarked

titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
# columns from training data we will use:

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



# initialise algorithm with default parameters

# n_estimators = num of trees

# min_samples_split = min num of rows we need to make a split

# min_samples_leaf = min # samples we can have a the place

## where a tree branch ends

alg = RandomForestClassifier(random_state=1,n_estimators=10, min_samples_split=2, min_samples_leaf=1)



kf = KFold(n_splits=3, random_state=1)

kf.get_n_splits(titanic.shape[0])

KFold(n_splits=3, random_state=1, shuffle=True)