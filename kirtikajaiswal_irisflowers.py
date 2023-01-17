import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas import read_csv

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
# load dataset

filename = "/kaggle/input/iris-flower-dataset/IRIS.csv"

data = read_csv(filename)

print(data.head(60))
# shape of data

data.shape
#data types of attributes

data.dtypes
#number of observations for each class 

data.groupby('species').size()
#statistical properties of attributes

data.describe()
#Correlations between each attribute

print(data.corr(method='pearson'))
#skweness of each attribute

data.skew()
#data visualization using density plots

data.plot(kind='density', sharex=False)

plt.show()
#data visualization using box and whisker plots

data.plot(kind='box', sharex=False, sharey=False)

plt.show()
#Scactter Matrix 

scatter_matrix(data)

plt.show()
# Split data in inputs and labels

array = data.values

inputs = array[:, 0:-1]

labels = array[:, -1]
# Scores of features

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=3)

fit = test.fit(inputs, labels)

print(fit.scores_)
# Best 3 features

features = fit.transform(inputs)

print(features)
# Rescale inputs

scaler = MinMaxScaler(feature_range=(0, 1))

rescaled_inputs = scaler.fit_transform(features)

print(rescaled_inputs[0:5, :])
# Standardize data to mean - 0 and std - 1

scaler = StandardScaler().fit(rescaled_inputs)

rescaled_inputs = scaler.transform(rescaled_inputs)

print(rescaled_inputs[0:5, :])
# Split data into train and test sets

input_train, input_test, label_train, label_test = train_test_split(rescaled_inputs, labels, test_size=0.33, random_state=7, shuffle=True)
# train the model

model = LinearDiscriminantAnalysis()

clf = model.fit(input_train, label_train)

# Predictions using model

predictions = model.predict(input_test)

print(accuracy_score(label_test, predictions))
# Confusion Matrix

print(confusion_matrix(label_test, predictions))
#Classification Matrix

print(classification_report(label_test, predictions))