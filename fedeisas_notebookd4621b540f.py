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
titanic_train = pd.read_csv('../input/train.csv')
titanic_train.describe()
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())
titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 0

titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 1
titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")

titanic_train.loc[titanic_train["Embarked"] == "S", "Embarked"] = 0

titanic_train.loc[titanic_train["Embarked"] == "C", "Embarked"] = 1

titanic_train.loc[titanic_train["Embarked"] == "Q", "Embarked"] = 2