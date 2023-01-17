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
data = pd.DataFrame(dict(x1=[1, 1, 0, 0], x2=[1, 0, 1, 0], result=[1, 1, 1, 0]))

data
x_train = np.array(data.loc[:, ['x1', 'x2']])

y_train = np.array(data.loc[:, ['result']])
def initialize_parameters():

    weights = np.random.rand(2,1)

    bias = np.random.rand()

    return weights, bias



def sigmoid(x):

    x = 1/(1+np.exp(-x)) 

    return x



def forward_backward_propagation(x_train, y_train, weights, bias):

#Forward Propagation

    y = np.dot(x_train, weights) + bias

    out_y = sigmoid(y)

    

#Error Function (Or Cost Function)

    #total_error = np.sum((y_train - out_y)**2) / (2 * y_train.shape[0])



#Backward Propagation

    derivative_weights = -( np.dot( x_train.T, (y_train-out_y) * out_y*(1-out_y)) ) / y_train.shape[0]

    derivative_bias = -np.sum((y_train-out_y) * out_y*(1-out_y)) / y_train.shape[0]

    return derivative_weights, derivative_bias

    

def update_parameters(x_train, y_train, weights, bias, learning_rate, number_of_iteration):  

    for i in range(number_of_iteration):       

        derivative_weights, derivative_bias = forward_backward_propagation(x_train, y_train, weights, bias)

        weights = weights - learning_rate * derivative_weights

        bias = bias - learning_rate * derivative_bias

    return weights, bias



def predict(weights, bias, x_test):

    prediction = sigmoid(np.dot(x_test, weights) + bias)

    for i in range(prediction.shape[0]):        

        if prediction[i] > 0.5:

            prediction[i] = 1

        else:

            prediction[i] = 0

    return prediction

        



def implement(x_train, y_train, learning_rate, number_of_iteration):

    weights, bias = initialize_parameters()

    weights, bias = update_parameters(x_train, y_train, weights, bias, learning_rate, number_of_iteration)

     

    prediction = predict(weights, bias, x_train)

    acc = 100 - ( (np.sum(np.abs(prediction-y_train)) / y_train.size) * 100 )

    

    params = dict(acc=acc, w=weights, b=bias)

    return prediction, params
#Implement with selected number of iteration

prediction, params = implement(x_train, y_train, learning_rate=0.3, number_of_iteration=2000)

print("Expected Values: ", y_train.T)

print("Results : ", prediction.T.astype(int))
#Defining big range of data



def fun(x1, x2, w, b):

    return sigmoid(x1 * w[0] + x2 * w[1] + b)



x1 = x2 = np.arange(0, 1.1, 0.1)

X1, X2 = np.meshgrid(x1, x2)



Z_sigmoid = np.array([fun(x1,x2,params["w"],params["b"]) for x1,x2 in zip(np.ravel(X1), np.ravel(X2))])

Z_logistic = [1 if Z_sigmoid[i]>0.5 else 0 for i in range(Z_sigmoid.size)]



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

    name = "OutputY = (sigmoid(w1*x1 + w2*x2 + b))",

    color='#FFB6C1',

    opacity=0.9

)



trace3 = go.Mesh3d(

    x=X1.ravel(), 

    y=X2.ravel(), 

    z=Z_logistic,

    name = "Logistic Regression(OutputY)",

    color='#FF5376',

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

        xaxis = dict(title='X1'),

        yaxis = dict(title='X2'),

        zaxis = dict(title='OUTPUT'),

    ),

)

    

fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

iplot(fig)