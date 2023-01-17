# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
passengers = pd.read_csv("../input/train.csv")



passengers.head()
passengers.drop(["PassengerId", "Name", "Ticket"], axis = 1, inplace = True)

passengers.head()
passengers["Relatives"] = passengers["SibSp"] + passengers["Parch"]

passengers.head(20)
passengers.drop(["SibSp", "Parch"], axis = 1, inplace = True)
passengers.head()
passengers.drop("Cabin", axis = 1, inplace = True)

passengers.head()
passengers.isna().sum()
passengers.head()
passengers.shape
passengers.dropna(subset = ["Embarked"], axis = 0, inplace = True)

passengers.shape
passengers.isna().sum()
passengers.head()
counts = passengers["Age"][passengers["Survived"] == 1 ]



counts.plot.hist(bins = 25)



counts2 = passengers["Age"][passengers["Survived"] == 0 ]



counts2.plot.hist(bins = 25)
passengers.head()
passengers["Child"] = passengers["Age"] <= 5

passengers[passengers["Child"] == False].head(30)
passengers.drop("Age", axis = 1, inplace = True)

passengers.head()

passengers.head()
passengers["Child"] = passengers["Child"].astype(int)

passengers.head(10)
passengers["Sex"] = (passengers["Sex"] == "male")
passengers.head()
passengers["Sex"] = passengers["Sex"].astype(int)

passengers.head()
def change(x):

    if x == "S":

        return 0

    elif x == "C":

        return 1

    else:

        return 2



passengers["Embarked_num"] = passengers["Embarked"].apply(change)

        

passengers.head(5)
passengers.drop("Embarked", axis = 1, inplace = True)

passengers.head()
from sklearn import preprocessing



passengers["standardized_Fare"] = preprocessing.scale(passengers["Fare"])
passengers.head()
passengers.drop("Fare", axis = 1, inplace = True)

passengers.head()
passengers.head()
passengers["Low"] = (passengers["Pclass"] >= 3).astype(int)

passengers.drop("Pclass", axis = 1, inplace = True)

passengers.head()


#'High', 'Relatives', 'Child', 'Embarked_num',

cols_list = ['Sex', 'standardized_Fare']

for col in cols_list:

    sns.violinplot(passengers.Survived, col, data = passengers)

    plt.show()
from sklearn.model_selection import train_test_split



X = passengers[['Sex', 'standardized_Fare',"Child"]]

y = passengers.Survived



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.80, test_size=0.20,

                                                      random_state=0)

X_valid.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier





# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = KNeighborsClassifier(n_neighbors = 10)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return accuracy_score(y_valid, preds)



score_dataset(X_train, X_valid, y_train, y_valid)