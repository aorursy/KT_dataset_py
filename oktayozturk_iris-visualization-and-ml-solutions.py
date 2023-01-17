# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import matplotlib.pyplot as plt
import seaborn as sns

# Importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing Data
data = pd.read_csv("../input/Iris.csv")
# Showing first five columns
data.head()
# Showing last five columns
data.tail()
# Checking Null Values
data.isnull().sum()
# We need to drop useless columns
data.drop(["Id"], axis = 1, inplace = True)
# Statistics Features
data.describe()
sns.jointplot(data.loc[:,'SepalLengthCm'], data.loc[:,'PetalLengthCm'], kind="regg", color="#ce1414")
sns.set(style="white")
df = data.loc[:,['SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=4)
# Histogram
# bins = number of bar in figure
data.SepalLengthCm.plot(kind = 'hist', bins = 50, figsize = (15,15))
data.SepalWidthCm.plot(kind = 'hist', bins = 50, figsize = (15,15))
data.PetalLengthCm.plot(kind = 'hist', bins = 50, figsize = (15,15))
data.PetalWidthCm.plot(kind = 'hist', bins = 50, figsize = (15,15))
plt.show()
# Correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# Create a trace
trace = go.Scatter(
    x = data.SepalLengthCm,
    y = data.SepalWidthCm,
    mode = 'markers'
)

data_2 = [trace]
fig = dict(data = data_2)
iplot(fig)
# Trace 2
trace1 = go.Scatter(
    x  = data.PetalLengthCm,
    y  = data.PetalWidthCm,
    mode = 'markers',
    marker = dict(
        size = 16,
        colorscale = 'Viridis',
        showscale = True
    )
)

data_1 = [trace1]
fig = dict(data = data_1)
iplot(fig)
trace2 = go.Box(
    y = data.PetalLengthCm
)

trace3 = go.Box(
    y = data.PetalWidthCm
)

data_2 = [trace2, trace3]
fig = dict(data = data_2)
iplot(fig)
# Split into train and test
# The attribute test_size = 0.2 splits the data into 80% and 20% ratio. train = 80% and test = 20%
train, test = train_test_split(data, test_size = 0.3)
# Four parameter going to help predict our main subject
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# Output our training data
train_x = train.Species
test_Y = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# Output our test data
test_y = test.Species
# Implement our regression
model = LogisticRegression()
# Fitting the model
model.fit(train_X,train_x)
# Predict the fitting data
prediction = model.predict(test_Y)
print('The accuracy is', metrics.accuracy_score(prediction,test_y))
# Implementing SVM
model = svm.SVC()
# Fitting the model
model.fit(train_X,train_x)
# Predict the fitting data
prediction = model.predict(test_Y) 
print('The accuracy is:',metrics.accuracy_score(prediction,test_y))
# Implementing Decision Tree Classifier
model = DecisionTreeClassifier()
# Fitting the model
model.fit(train_X,train_x)
prediction = model.predict(test_Y)
print('The accuracy is',metrics.accuracy_score(prediction,test_y))
# This examines 3 neighbours
model = KNeighborsClassifier(n_neighbors=3) 
# Fitting the model
model.fit(train_X,train_x)
prediction = model.predict(test_Y)
print('The accuracy is',metrics.accuracy_score(prediction,test_y))
# Find Best K Value
score_list = []
for each in range(1,50):
    knn_2 = KNeighborsClassifier(n_neighbors = each)
    knn_2.fit(train_X, train_x)
    score_list.append(knn_2.score(test_Y,test_y))

plt.plot(range(1,50), score_list)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.show()