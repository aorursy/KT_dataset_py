import matplotlib

import numpy as np

import pandas as pd

import os

import sklearn

from sklearn import metrics

from sklearn.preprocessing import StandardScaler 

from pandas import DataFrame

from pandas import concat

from pandas import read_csv

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import seaborn as sns
# Breast Cancer dataset

# Citation: Dr. William H. Wolberg, University of Wisconsin Hospitals, Madison 

# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)



# Read the dataset (Note that the CSV provided for this demo has rows with the missing data removed)

df =  pd.read_csv('../input/breastcancer.csv', header=0)



# Take a look at the structure of the file

df.head(n=10)
# Drop Id column not used in analysis

df.drop(['Id'], 1, inplace=True)



# Label encoding Target variable

encoder = LabelEncoder()

df['Class'] = encoder.fit_transform(df['Class'])
# Check the result of the transform

df.head(n=6)
X = np.array(df.drop(['Class'], axis=1))

y = np.array(df['Class'])



# Scale the data. We will use the same scaler later for scoring function

scaler = StandardScaler().fit(X)

X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
print('Train Score:',classifier.score(X_train,y_train))

print('Test Scxore:',classifier.score(X_test,y_test))