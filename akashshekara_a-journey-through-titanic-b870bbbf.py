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
train_df = DataFrame.from_csv("../input/train.csv", header = 0)

test_df = DataFrame.from_csv("../input/test.csv", header = 0)
train_df.head()
train_df = train_df.drop(["Name","Ticket"], axis=1)

test_df = test_df.drop(["Name","Ticket"], axis=1)
#Embarked



countS = 0

countC = 0

countQ = 0



countS = train_df["Embarked"][train_df["Embarked"] == "S"]

countC = train_df["Embarked"][train_df["Embarked"] == "C"]

countQ = train_df["Embarked"][train_df["Embarked"] == "Q"]



print("CountS: " + countS + "\nCountC" + countC + "\nCountQ" + countQ)