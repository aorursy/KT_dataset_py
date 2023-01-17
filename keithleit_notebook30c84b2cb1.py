import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error,mean_absolute_error

import matplotlib.pyplot as plt

#print(check_output(["ls", "../input"]).decode("utf8"))

data=pd.read_csv("../input/train.csv")

real_test=pd.read_csv("../input/test.csv")
print(train.shape, train.columns)

train,test=train_test_split(data)
clf=LogisticRegression()

tree=DecisionTreeClassifier()

rf=RandomForestClassifier()
print(train.notnull().sum())