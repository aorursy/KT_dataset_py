# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from sklearn.tree import DecisionTreeRegressor

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Admission_Predict.csv')
df.head()
#Scatter Plots Probably makes more sense for continuous factors like Scores
factors = df.columns
factors = factors[1:-1]

plt.figure(figsize=(20, 20))
for i in range(len(factors)):
    plt.subplot(3,3,i+1)
    plt.scatter(df[factors[i]],df['Chance of Admit '])
    plt.title(factors[i])
#For Discrete factors bar plots of the averages acceptance percent works
#And lets put a line to see what the median student's chances are
factors = df.columns
factors = factors[[3,4,5,7]]
median = df['Chance of Admit '].median()

plt.figure(figsize=(20, 10))
for j in range(len(factors)):
    plt.subplot(2,2,j+1)
    values = df[factors[j]].unique()
    ser = pd.Series(range(len(values)), index = values, dtype='float64')
    for i in range(len(values)):
        ser[values[i]] = df[df[factors[j]]==values[i]]['Chance of Admit '].mean()
    ser = ser.sort_index()
    plt.bar(ser.index,ser.values,width=0.3)
    plt.title(factors[j])
    plt.plot([0,len(values)],[median,median],'k-', lw=1,dashes=[2, 2])
    
#Making the model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

categories = ['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']
model = DecisionTreeRegressor(random_state=0)

X = df[categories]
y = df['Chance of Admit ']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.25)

model.fit(train_X,train_y)

#Small validation
predicted = model.predict(val_X)
mean_absolute_error(val_y, predicted)

#Lets make a graph for overfitting vs underfitting
leaf_nodes = range(2,50)
series = pd.Series(range(len(leaf_nodes)),index=leaf_nodes,dtype='float64')

#Defining Data
X = df[categories]
y = df['Chance of Admit ']
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.25)

for i in leaf_nodes:
    model = DecisionTreeRegressor(max_leaf_nodes = i, random_state=0)
    model.fit(train_X,train_y)
    predicted = model.predict(val_X)
    series[i] = mean_absolute_error(val_y, predicted)

plt.scatter(series.index,series.values)
print("Minimum is " + str(series.idxmin()))
#Final Model
model = DecisionTreeRegressor(max_leaf_nodes = 10, random_state=0)
model.fit(train_X,train_y)
predicted = model.predict(val_X)
mae = mean_absolute_error(val_y, predicted)

print(mae)
