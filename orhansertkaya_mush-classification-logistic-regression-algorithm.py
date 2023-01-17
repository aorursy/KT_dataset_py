# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/mushrooms.csv')
data.info()  # Display the content of data
# shape gives number of rows and columns in a tuple
data.shape
data.head()
data.tail()
data.describe()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data = data.apply(label_encoder.fit_transform)
data.dtypes
data.head()
data.sample(5)
y = data['class'].values
x_data = data.drop(['class'], axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
x.isnull().sum() #Indicates values not defined in our data
#as you see there are 8124 NaN values
#we drop veil_type feature from data
x.drop(["veil-type"],axis=1,inplace=True) 
x.isnull().sum().sum()  #Indicates sum of values in our data
print(x.shape)
print(y.shape)
x.head(10)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b
def sigmoid(z):
    
    y_head = 1/(1 + np.exp(-z))
    
    return y_head
def forward_backward_propagation(w, b, x_train, y_train):
    #forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    return cost,gradients
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    
    cost_list = []
    index = []
    
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w, b, x_train, y_train)
        
        # lets update
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        
        cost_list.append(cost)
        index.append(i)
        
    parameters = {"weight":w,"bias":b}
    
    # Creating trace1
    trace1 = go.Scatter(
                    x = index,
                    y = cost_list,
                    mode = "lines",
                    name = "Cost",
                    marker = dict(color = 'rgba(160, 112, 2, 0.8)'),
                    text= index)
    data = [trace1]
    
    layout = dict(title = 'Number of Cost and Cost Values',
              xaxis= dict(title= 'Number of Cost',ticklen= 10,zeroline= True))
    
    fig = dict(data = data, layout = layout)
    iplot(fig)
    
    # Creating trace1
    trace1 = go.Scatter(
                    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
                    y = parameters['weight'].reshape(x_train.shape[0]), #print(parameters['weight'].reshape(21).shape)
                    mode = "lines",
                    name = "Weight",
                    marker = dict(color = 'rgba(255, 77, 77, 1)'),
                    text=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])    
    # Creating trace2
    trace2 = go.Scatter(
                    x = [1],
                    y = np.array(parameters['bias']),
                    mode = "markers+text",
                    name = "Bias",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= ["Bias Value"])    
    
    data = [trace1,trace2]
    
    layout = dict(title = 'Weights and Bias Values',
              xaxis= dict(title= 'Number of Weights',ticklen= 10,zeroline= True))
    
    fig = dict(data = data, layout = layout)
    iplot(fig)
    
    return parameters,gradients,cost_list
                
def predict(w, b, x_test):
    
    # x_test is a input for forward propagation
    z = np.dot(w.T,x_test) + b
    y_head = sigmoid(z)
    y_prediction = np.zeros((1,x_test.shape[1]))
    
    # if z is bigger than 0.5, our prediction is one (y_head=1)
    # if z is smaller than 0.5, our prediction is zero (y_head=0)
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    
    return y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # initialize
    dimension = x_train.shape[0] #that is 21
    w,b = initialize_weights_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    
    #Print Train Accuracy
    train_accuracy = (100 - np.mean(np.abs(y_prediction_train - y_train)) * 100)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    #Print Test Accuracy
    test_accuracy = (100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
    # create trace1 
    trace1 = go.Bar(
                x = np.array("Train Accuracy"),
                y = np.array(train_accuracy),
                name = "Train Accuracy",
                marker = dict(color ='rgba(1, 255, 128, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5))
                    )
    # create trace2 
    trace2 = go.Bar(
                x = np.array("Test Accuracy"),
                y = np.array(test_accuracy),
                name = "Test Accuracy",
                marker = dict(color ='rgba(1, 128, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5))
                    )

    data = [trace1,trace2]
    layout = go.Layout(barmode = "group")
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, num_iterations = 500)
from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression()
log_reg_model.fit(x_train.T,y_train.T)

#Print Train Accuracy
train_accuracy_sk = (log_reg_model.score(x_train.T,y_train.T))*100
print("train accuracy {}".format(log_reg_model.score(x_train.T,y_train.T)*100))
test_accuracy_sk = (log_reg_model.score(x_test.T,y_test.T))*100
print("test accuracy {}".format(log_reg_model.score(x_test.T,y_test.T)*100))

# create trace1 
trace1 = go.Bar(
         x = np.array("Train Accuracy"),
         y = np.array(train_accuracy_sk),
         name = "Train Accuracy",
         marker = dict(color ='rgba(255, 77, 77, 1)',
         line=dict(color='rgb(0,0,0)',width=1.5))
                    )
    # create trace2 
trace2 = go.Bar(
         x = np.array("Test Accuracy"),
         y = np.array(test_accuracy_sk),
         name = "Test Accuracy",
         marker = dict(color ='rgba(36, 255, 222, 1)',
         line=dict(color='rgb(0,0,0)',width=1.5))
                    )

data = [trace1,trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
y_pred = log_reg_model.predict(x_test.T)
y_pred
y_pred = log_reg_model.predict(x_test.T)
y_true = y_test.T

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot=True, linewidth=0.5, fmt=".0f",  cmap='RdPu', ax = ax)
plt.xlabel = ("y_pred")
plt.ylabe = ("y_true")
plt.show()