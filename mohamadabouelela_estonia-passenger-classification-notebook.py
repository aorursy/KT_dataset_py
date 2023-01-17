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
# Prepare required tools



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



from sklearn.linear_model import RidgeClassifier

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
# read CSV file



estonia = pd.read_csv ("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")

estonia.head()
estonia.tail()
estonia["Country"].value_counts()
estonia["Category"].value_counts()
estonia.Sex.value_counts()
# EDA _ Exploratory Data Analysis



estonia.info()
# check for any null features

estonia.isna().sum()
# describe dataset

estonia.describe()

# how many (survived = 1) and how many (did not survive = 0) 

estonia["Survived"].value_counts()
pd.crosstab(estonia.Sex, estonia.Survived)
pd.crosstab(estonia.Sex, estonia.Survived).plot.bar()

plt.title ("SEX VS Survival")

plt.ylabel ("Number of passengers")

plt.xticks (rotation = 0)

plt.legend (["Didn't Survive", "Survive"]);
pd.crosstab(estonia.Category, estonia.Survived)
pd.crosstab(estonia.Category, estonia.Survived).plot.bar()

plt.title ("Category VS Survival")

plt.ylabel ("Number os passengers")

plt.xticks (rotation = 0)

plt.legend (["Didn't survive", "Survive"]);
estonia.Age.plot.hist();
# plot Age & Country VS survival

plt.figure (figsize = (10, 6))

plt.scatter (estonia.Age[estonia.Survived ==1],

             estonia.Country[estonia.Survived ==1],

             c = "orange")

plt.title ("Age and Country VS Survival")

plt.legend (["Survive"])

plt.xlabel ("Age");
# plot Age & Country VS No Survival

plt.figure (figsize = (10, 6))

plt.scatter (estonia.Age[estonia.Survived ==0],

             estonia.Country[estonia.Survived ==0],

             c = "blue")

plt.title ("Age and Country VS No Survival")

plt.legend (["Didn't Survive"])

plt.xlabel ("Age");
# lets drop less important columns in our data set to prepare our correlation matrix

estonia.drop(["PassengerId", "Firstname", "Lastname", "Country"], axis=1,inplace = True)

estonia.head()
# change Sex and Category to numerical 

change_dict = {"Sex" : {"M" : 1,

                        "F" : 0},

              "Category" : {"P" : 1,

                            "C" : 0}}

estonia.replace(change_dict, inplace = True)

estonia.head()
# We can get the correlation matrix now 



corr_matrix = estonia.corr()

corr_matrix
# lets make the correlation matrix more visual

fig, ax = plt.subplots(figsize = (8, 6))

plt.ax = (sns.heatmap(corr_matrix,

                      cmap = "YlGnBu",

                      cbar = False,

                      annot = True,

                      fmt = ".2f"));
# Start Modeling 

# keep all featires as X and target as y

X = estonia.drop("Survived", axis = 1)

y = estonia["Survived"]
X
# split data to train and test 

np.random.seed (99)

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2)
# we are going to be testing on Three models



models = {"Ridge classifier" : RidgeClassifier(),

          "Random Forest" : RandomForestClassifier(),

          "KNN" : KNeighborsClassifier()}

def fit_and_score(models, X_train, X_test, y_train, y_test):

    models_score = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        models_score[name] = model.score(X_test, y_test)

    return models_score
models_score = fit_and_score(models, X_train, X_test, y_train, y_test)

models_score