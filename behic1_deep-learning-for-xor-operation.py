# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
data = pd.DataFrame(dict(x1=[1, 1, 0, 0], x2=[1, 0, 1, 0], result=[0, 1, 1, 0]))

x_train = np.array(data.loc[:, ['x1', 'x2']]).T

y_train = np.array(data.loc[:, ['result']]).T

data
# intialize parameters and layer sizes(Just giving some random numbers, you can change if you wish)

def initialize_parameters():

    parameters = {"weight1": np.array([[0.0001, 0.2343], [0.003451, 0.22113]]),

                  "bias1": np.array([[0.13511], [0.0989]]),

                  "weight2": np.array([[0.45641, 0.00099]]),

                  "bias2": np.array([[0.009]]),

                 }

    return parameters



# Sigmoid

def sigmoid(x):

    x = 1/(1+np.exp(-x))

    return x



# Cost

def cost_funtion(out_y, y_train):

    loss = -y_train*np.log(out_y)-(1-y_train)*np.log(1-out_y)

    cost = np.sum(loss)/y_train.shape[1]

    return cost



def forward_backward_propagation(parameters, x_train, y_train):

    #forward

    h = np.dot(parameters["weight1"],x_train) +parameters["bias1"]

    out_h = np.tanh(h)

    y = np.dot(parameters["weight2"],out_h) + parameters["bias2"]

    out_y = sigmoid(y)

    

    #cost

    cost = cost_funtion(out_y, y_train)

    

    #backward

    sigma_y = out_y-y_train 

    dW2 = np.dot(sigma_y,out_y.T)/y_train.shape[1]

    db2 = np.sum(sigma_y,axis =1,keepdims=True)/y_train.shape[1]

    

    sigma_h = np.dot(parameters["weight2"].T,sigma_y)*(1 - np.power(out_h, 2))

    dW1 = np.dot(sigma_h,x_train.T)/y_train.shape[1]

    db1 = np.sum(sigma_h,axis =1,keepdims=True)/y_train.shape[1]

    

    derivatives = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return derivatives, cost



# update parameters

def update_parameters(parameters, x_train, y_train, learning_rate, iteration):

    cost_list = []

    cost_index = []

    for i in range(iteration):

        derivatives,cost = forward_backward_propagation(parameters, x_train, y_train)

        parameters = {"weight1": parameters["weight1"] - learning_rate*derivatives["dweight1"],

                      "bias1": parameters["bias1"] - learning_rate*derivatives["dbias1"],

                      "weight2": parameters["weight2"] - learning_rate*derivatives["dweight2"],

                      "bias2": parameters["bias2"] - learning_rate*derivatives["dbias2"]}

        if i % 100 == 0:

            cost_list.append(cost)

            cost_index.append(i)    

    return parameters, cost_index,cost_list



# prediction

def predict(parameters, x_train):

    h = np.dot(parameters["weight1"],x_train) +parameters["bias1"]

    out_h = np.tanh(h)

    y = np.dot(parameters["weight2"],out_h) + parameters["bias2"]

    out_y = sigmoid(y)

    y_prediction = np.zeros((1,x_train.shape[1]))

    

    for i in range(out_y.shape[1]):

        if out_y[0,i]<= 0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

    return y_prediction



# 2 - Layer neural network

def implement(x_train, y_train, learning_rate, iteration):

    parameters = initialize_parameters()

    parameters, cost_index,cost_list = update_parameters(parameters, x_train, y_train, learning_rate, iteration)

    

    plt.plot(cost_index,cost_list)

    plt.title("Cost According To The Iterations")

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()    

        

    # predict

    y_prediction = predict(parameters,x_train)

    

    return parameters, y_prediction
parameters, y_prediction= implement(x_train, y_train, learning_rate=0.3, iteration=1200)



print("Expected Values: ", y_train)

print("Results : ", y_prediction.astype(int))
def fun(parameters, x_train):

    h = np.dot(parameters["weight1"],x_train) +parameters["bias1"]

    out_h = np.tanh(h)

    y = np.dot(parameters["weight2"],out_h) + parameters["bias2"]

    out_y = sigmoid(y)

    return out_y



x1 = x2 = np.arange(0, 1.1, 0.1)

X1, X2 = np.meshgrid(x1, x2)

INPUTS = np.array([np.ravel(X1), np.ravel(X2)])



Z_sigmoid = fun(parameters, INPUTS)

Z_logistic = [1 if Z_sigmoid[0,i]>0.5 else 0 for i in range(Z_sigmoid.size)]



surface_data = pd.DataFrame(dict(X1=X1.ravel(), X2=X2.ravel(), Z=Z_sigmoid.ravel()))
trace1 = go.Scatter3d(

    x=data.x1,

    y=data.x2,

    z=data.result,

    name = "Values",

    mode='markers',  

)

trace2 = go.Mesh3d(

    x=X1.ravel(), 

    y=X2.ravel(), 

    z=Z_sigmoid.ravel(),

    name = "OutputY (Sigmoid Function))",

    color='#FFB6C1',

    opacity=0.9

)

trace3 = go.Mesh3d(

    x=X1.ravel(), 

    y=X2.ravel(), 

    z=Z_logistic,

    name = "Logistic Regression(OutputY)",

    color='#008BF8',

    opacity=0.7

)



layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ), 

    scene=dict(

        camera=dict( eye=dict(x=1.0700, y=-1.700, z=0.6500) ),

        xaxis = dict(title='x1'),

        yaxis = dict(title='x2'),

        zaxis = dict(title='output'),

    ),

)    

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

iplot(fig)