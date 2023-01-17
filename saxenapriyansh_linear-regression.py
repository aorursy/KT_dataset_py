import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))
# about the dataset
# Goto: https://www.kaggle.com/baiazid/coursera-machine-learning-su-ex1

df = pd.read_csv('../input/week_1-ex_1.txt',header=None,names=('polpulation/1000','profit/1000'))
# The first column represents city population (* 10,000 to get the real value) 
# The second column represents profit of a food truck (* 10,000 to get the real value)

# First some context on the problem statement. Here we will implement linear regression with one variable 
# to predict profits for a food truck. Suppose you are the CEO of a restaurant franchise and are considering
# different cities for opening a new outlet. The chain already has trucks in various cities and you have data
# for profits and populations from the cities. The file ex1data1.txt (available under week 2's assignment material)
# contains the dataset for our linear regression exercise. The first column is the population of a city and the 
# second column is the profit of a food truck in that city. A negative value for profit indicates a loss.
print(df.head())
print(df.shape)
m=df.shape[0]
X = df.iloc[:,0]
y = df.iloc[:,1]
import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
# How to display images in ipython notebook
# Goto: https://stackoverflow.com/questions/11854847/how-can-i-display-an-image-from-a-file-in-jupyter-notebook
from IPython.display import Image, display

#listOfImageNames = ['C:/Users/saxenapriyansh/Desktop/linear_reg-1var-hypothesis.PNG',
#                    'C:/Users/saxenapriyansh/Desktop/linear_reg-1var-cost_function.PNG',
#                    'C:/Users/saxenapriyansh/Desktop/linear_reg-1var-gradient_descent.PNG']

#for imageName in listOfImageNames:
#    display(Image(filename=imageName))
# In the following lines, we add another dimension to our data to accommodate the intercept term 
# (the reason for doing this is explained in the videos). We also initialize the initial parameters
# theta to 0 and the learning rate alpha to 0.01.
# newaxis is used to increase the dimension of the existing array by one more dimension, when used once. Thus,
# 1D array will become 2D array
# 2D array will become 3D array
# 3D array will become 4D array
# and so on.

# Goto: https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
import numpy as np
X = X[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
# hstach - Stack arrays in sequence horizontally (column wise)
# for more info on hstack
# Goto: https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html
X = np.hstack((np.ones((m,1)), X))
# New shape of X,y 
print(X.shape)
print(X[0:5,:])
print("***--***")
print(y.shape)
print(y[0:5,:])
# for more info on np.dot
# Goto: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.dot.html

# for more info on numpy.square
# Goto: https://docs.scipy.org/doc/numpy/reference/generated/numpy.square.html

# for more info on numpy.sum
# Goto: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.sum.html#numpy.sum

def computeCost(X, y, theta):
    temp = np.dot(X,theta) - y
    return np.sum(np.square(temp))/(2*m)
J = computeCost(X,y,theta)
print(J)
# Initial value of theta :
print(theta)
# more about ' - ' : Goto: https://hackernoon.com/understanding-the-underscore-of-python-309d1a029edc
def gradientDescent(theta, X, y, aplha, iterations):    
    
    # theta_vals, J_vals stores the intermediate values of theta and J during optimiztion usng gradient descent
    theta_vals=np.full([1,2],0)
    J_vals = np.full([1],computeCost(X,y,theta))
    
    for _ in range(iterations):
        theta = theta - (aplha/m)*np.dot(X.T,(np.dot(X, theta) - y))        
        theta_vals = np.vstack((theta_vals,theta.T))
        J_vals = np.vstack((J_vals,computeCost(X, y, theta)))
    return [theta_vals,J_vals]


(theta_vals,J_vals) = gradientDescent(theta, X, y, alpha, iterations)
theta = np.reshape(theta_vals[-1],[2,1])
# theta after optimizing
print("Theta: ",theta)
print(theta)
# After finding optimum value of theta, find the new cost
J = computeCost(X,y,theta)
print(J)
plt.scatter(X[:,1], y) # plt.scatter plots only the points
plt.plot(X[:,1], np.dot(X, theta),'r') # plt.plot draws a line
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
# Plotly - one stop for all plots, visualizations, animations etc
# Goto: https://plot.ly/python/

import plotly
plotly.tools.set_credentials_file(username='enter_your_details_here', api_key='8tUzNDbRW7G2wlH4js1P')

import plotly.plotly as py
import plotly.graph_objs as go

x_val = theta_vals[:,0]
y_val = theta_vals[:,1]  
z_val = J_vals[:,0]                 

line = go.Scatter3d(x=x_val, y=y_val, z=z_val, marker=dict(
        size=4,
        color=z_val,
        colorscale='Viridis',
    ),
    line=dict(
        color='#1f77b4',
        width=1
    ))
data = [line]


layout = go.Layout(
    title='Theta values vs Cost values',
    scene=dict(
        xaxis=dict(
            title = 'theta 0',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            title = 'theta 1',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            title = 'Cost',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        )
    )
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='jupyter-parametric_plot')
# Problem context: Suppose you are selling your house and you want to know what a good market price would be. 
# One way to do this is to first collect information on recent houses sold and make a model of housing prices. 
# Your job is to predict housing prices based on other variables.

#The file ex1data2.txt((available under week 2’s assignment material)) contains a training set of housing prices
# in Portland, Oregon. The first column is the size of the house (in square feet), the second column is the number 
# of bedrooms, and the third column is the price of the house.

# You already have the necessary infrastructure which we built in our previous section that can be easily 
# applied to this section as well. Here we will just use the equations which we made in the above section.
df = pd.read_csv('../input/week_2-ex_1.txt',header=None,names=('size','bedrooms','price(in $)'))
print(df.head())
print(df.shape)
m=df.shape[0]
n=df.shape[1]
X = df.iloc[:,0:2]
y = df.iloc[:,2]
# Plotly - one stop for all plots, visualizations, animations etc
# Goto: https://plot.ly/python/

import plotly
plotly.tools.set_credentials_file(username='enter_your_details_here', api_key='8tUzNDbRW7G2wlH4js1P')

import plotly.plotly as py
import plotly.graph_objs as go

x_val = X['size']
y_val = X['bedrooms']
z_val = y[:]                 

line = go.Scatter3d(x=x_val, y=y_val, z=z_val, mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    ))
data = [line]


layout = go.Layout(
    title='Theta values vs Cost values',
    scene=dict(
        xaxis=dict(
            title = 'size',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            title = 'bedrooms',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            title = 'price',
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        )
    )
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='jupyter-parametric_plot')
# How to display images in ipython notebook
# Goto: https://stackoverflow.com/questions/11854847/how-can-i-display-an-image-from-a-file-in-jupyter-notebook
from IPython.display import Image, display

#listOfImageNames = ['C:/Users/saxenapriyansh/Desktop/linear_reg-multivar-basics.PNG',
#                    'C:/Users/saxenapriyansh/Desktop/linear_reg-multivar-hypothesis.PNG',
#                    'C:/Users/saxenapriyansh/Desktop/linear_reg-multivar-cost_function.PNG',
#                    'C:/Users/saxenapriyansh/Desktop/linear_reg-multivar-gradient_descent.PNG',
#                    'C:/Users/saxenapriyansh/Desktop/linear_reg-multivar-feature_scaling_and_normalization.PNG',
#                    'C:/Users/saxenapriyansh/Desktop/linear_reg-multivar-normal_equation.PNG',
#                    'C:/Users/saxenapriyansh/Desktop/linear_reg-multivar-comparision_btw_gradient_descent_and_normal_eqation.PNG',]

#for imageName in listOfImageNames:
#    display(Image(filename=imageName))
#    print("--") 
X = (X - np.mean(X))/np.std(X)

X = np.column_stack((X['size'],X['bedrooms']))
X = np.hstack((np.full([m,1],1), X))
print(X.shape)
print(X[:5,:])

y = y[:,np.newaxis]
print(y.shape)
print(y[:5,])
theta = np.full([n,1],0)
alpha = 0.01
iterations = 50000
def computeCost(X, y, theta):
    temp = np.dot(X,theta) - y
    return np.sum(np.square(temp))/(2*m)
J = computeCost(X,y,theta)
print(J)
# more about ' - ' : Goto: https://hackernoon.com/understanding-the-underscore-of-python-309d1a029edc
def gradientDescent(theta, X, y, aplha, iterations):        
    # theta_vals, J_vals stores the intermediate values of theta and J during optimiztion usng gradient descent
    theta_vals=np.full([1,n],0)
    J_vals = np.full([1],computeCost(X,y,theta))
    
    for i in range(iterations):
        theta = theta - (aplha/m)*np.dot(X.T,(np.dot(X, theta) - y))    
        theta_vals = np.vstack((theta_vals,theta.T))
        J_vals = np.vstack((J_vals,computeCost(X, y, theta)))
    return [theta_vals,J_vals]
(theta_vals,J_vals) = gradientDescent(theta, X, y, alpha, iterations)
theta = np.reshape(theta_vals[-1],[n,1])
# theta after optimizing
print("Theta: ",theta)
J = computeCost(X, y, theta)
print(J)
import plotly.plotly as py
import plotly.graph_objs as go

# Create a trace
trace = go.Scatter(
    x = np.arange(1,50002,1),
    y = J_vals[:,0]
)
data = [trace]
py.iplot(data, filename='basic-line')
theta.shape
# Goto: https://youtu.be/N4d_9GQ9QFc
# https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression

theta_n = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
print(theta_n)
J = computeCost(X ,y,theta_n)
J
df = pd.read_csv('../input/week_2-ex_1.txt',header=None,names=('size','bedrooms','price(in $)'))
X = df.iloc[:,0:2]
y = df.iloc[:,2]
m=df.shape[0]  # number of examples
n=df.shape[1]  # number of features
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)
print(reg.score(X,y))
print(reg.get_params())
# finding the cost
y_pred = reg.predict(X)
def computeCost(y_pred, y_true):
    return np.sum(np.square(y_pred - y_true))/(2*m)
    
J = computeCost(y_pred, y)
print(J)
