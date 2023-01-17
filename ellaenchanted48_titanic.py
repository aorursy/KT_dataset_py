# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

## get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# preview the data

titanic_df.head()

titanic_df.info()

print("----------------------------")

test_df.info()

import numpy as np

from random import randint

myMatrix=[]

for i in range(418):

    if (titanic_df["Sex"][i+1]=="female"):

        passengerLabel=1

    else:

        passengerLabel=0

    myMatrix.append(passengerLabel)

myArray=np.array(myMatrix)

Y_pred=myArray.transpose()

print(Y_pred)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)