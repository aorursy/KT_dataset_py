!pip install sweetviz
# Standart Inputs

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# EDA tool

import sweetviz



# Used to display Html file for a later step

import IPython



# Model Selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



# Models

from sklearn.linear_model import LogisticRegression
data = pd.read_csv("../input/breast-cancer-wisconsin-benign-or-malignant/tumor.csv")
# Check data size and view first 10 rows

print(f"The data has {data.shape[1]} features and {data.shape[0]} rows\n")

data.head(10)
# Now we drop the id column since it's doesn't contain valuable information

data.drop([data.columns[0]], axis= 1, inplace= True)
# Check for null values

pd.isnull(data).sum()

# The data has no null values
# Check types of columns

data.dtypes
# Our classes are encoded as '2' and '4' for malignant and benign

# these two lines instead codes them at 0 and 1 for same classes

data['Class'].loc[data['Class'] == 2] = 0

data['Class'].loc[data['Class'] == 4] = 1
# View Distributions of each feature

data.hist( grid = False,yrot= 30, figsize=(16,12))

plt.show()
# Let's analyze the full data set with the amazing SweetViz tool

report = sweetviz.analyze(data, target_feat='Class')
# Since the sweetviz return an html file that's saved we need to display it here using the IPython command

IPython.display.IFrame(src='SWEETVIZ_REPORT.html', width=1080, height=600)
# Let's have a closer look at the two columns we noticed to have high correlation

# we'll check the percentage of malignant tumor to a cell size higher than 5

# you too can change the cell shape to view different insights

cellShape = 5

mask = data['Uniformity of Cell Shape'] > cellShape

filteredData = data[mask]

malignantTumors = filteredData.loc[data['Class'] == 1]

print(f"{int(malignantTumors.shape[0] / filteredData.shape[0] * 100)}% of {filteredData.shape[0]} tumors with uniformity of cell shape higher than {cellShape} are malignant")



# Let's check cell size too

cellSize = 4

mask = data['Uniformity of Cell Size'] > cellSize

filteredData = data[mask]

print(f"{int(malignantTumors.shape[0] / filteredData.shape[0] * 100)}% of {filteredData.shape[0]} tumors with uniformity of cell size higher than {cellSize} are malignant")

# Encode labels to strings

labels = data['Class']

labels[labels==1] = 'Malignant'

labels[labels==0] = 'Benign'



# Plot relation between the two columns labeld by type of tumor

plt.figure(figsize=(7,7))

sns.set_style('dark')

sns.scatterplot(data=data, x='Uniformity of Cell Shape', y= 'Uniformity of Cell Size',

                       hue= labels,

                       s=100)

plt.show()
# Adding name of columns to variables for easier use

uCellSize = data.columns[1]

uCellShape = data.columns[2]

print(f"{uCellSize}\n{uCellShape}")
# Split data to train and target

X , Y = data[[uCellSize,uCellShape]] , data['Class']

X.head(2)
# Split data into train and validation data

# this can give us intuition on how the model would do in data it hasn't been exposed to

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

print(f"Train size {X_train.shape[0]} rows\nTest Size {X_test.shape[0]} rows")
n_iterations = 100
# Prepare model

logisticModel = LogisticRegression(max_iter = n_iterations, multi_class= 'ovr', class_weight='balanced')
# Cross validation score

scores = cross_val_score(logisticModel,X_train,Y_train, cv= 3)

print(f"Average Cross Validation Score: {sum(scores) / 3}")
# fit on train data

logisticModel.fit(X_train,Y_train)

# socre on validation data

logisticModel.score(X_test, Y_test)
X_train, X_test, Y_train, Y_test = train_test_split(data.drop(['Class'], axis = 1), data['Class'], test_size=0.1)

print(f"Train size {X_train.shape[0]} rows\nTest Size {X_test.shape[0]} rows")
allFeaturesModel = LogisticRegression(max_iter= 100, multi_class='ovr',class_weight='balanced')
scores = cross_val_score(allFeaturesModel,X_train, Y_train, cv= 3)

print(f"Average Cross Validation Score: {sum(scores) / 3}")
# fit on train data

allFeaturesModel.fit(X_train,Y_train)

# socre on validation data

allFeaturesModel.score(X_test, Y_test)