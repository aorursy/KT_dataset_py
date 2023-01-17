from IPython.display import YouTubeVideo



YouTubeVideo('yIYKR4sgzI8', width=800, height=300)
from IPython.display import Image

import os

Image("../input/week-3-images/Logistic-Regression-Classification-1.jpeg")
"""

Sample data

"""



scores=[[1],[1],[2],[2],[3],[3],[3],[4],[4],[5],[6],[6],[7],[7],[8],[8],[8],[9],[9],[10]]

passed= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
#Type your code here-

"""

Draw a scatter plot with sample data

"""



from matplotlib import pyplot as plt

%matplotlib inline

 

plt.scatter(scores, passed, color='r')

plt.xlabel("scores")

plt.ylabel("passed")
"""

Linear regression fitting

"""



from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(scores, passed)

model.coef_, model.intercept_
"""

Draw the plot after fitting process

"""



import numpy as np



x = np.linspace(-2,12,100)



plt.plot(x, model.coef_[0] * x + model.intercept_)

plt.scatter(scores, passed, color='r')

plt.xlabel("scores")

plt.ylabel("passed")
#!ls ../input/week-3-images/

Image("../input/week-3-images/Untitled.jpg", width="800")
"""

Sigmoid function

"""



z = np.linspace(-12, 12, 100) # Generate equidistant x values for easy drawing

sigmoid = 1 / (1 + np.exp(-z))

plt.plot(z, sigmoid)

plt.xlabel("z")

plt.ylabel("y")
"""

Logistic Regression Model

"""



def sigmoid(z):

    sigmoid = 1 / (1 + np.exp(-z))

    return sigmoid
YouTubeVideo('fr7dfyfB7mI', width=800, height=300)
"""

Logarithmic Loss Function

"""



def loss(h, y):

    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    return loss
Image("../input/week-3-images/Logistic Regression Classification 4.jpeg", width="800")
YouTubeVideo('sDv4f4s2SB8', width=800, height=300) 
"""

Calculate gradient

"""



def gradient(X, h, y):

    gradient = np.dot(X.T, (h - y)) / y.shape[0]

    return gradient
"""

Load dataset

"""



import pandas as pd



df = pd.read_csv("../input/week-3-dataset/Logistic-Regression-Classification-data.csv", header=0) # Load dataset

df.head() # Preview first 5 rows of data
"""

Plot the data distribution

"""



plt.figure(figsize=(10, 6))

plt.scatter(df['X0'],df['X1'], c=df['Y'])
"""

Complete logistic regression process

"""



#Sigmoid distribution function

def sigmoid(z):

    sigmoid = 1 / (1 + np.exp(-z))

    return sigmoid



#Loss function

def loss(h, y):

    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    return loss



#Calculate the gradient

def gradient(X, h, y):

    gradient = np.dot(X.T, (h - y)) / y.shape[0]

    return gradient



#Logistic regression process

def Logistic_Regression(x, y, lr, num_iter):

    intercept = np.ones((x.shape[0], 1)) # Initialize intercept as 1

    x = np.concatenate((intercept, x), axis=1)

    w = np.zeros(x.shape[1]) # Initialize parameters as 1

    

    for i in range(num_iter): # Gradient descent iterations

        z = np.dot(x, w) # Linear function

        h = sigmoid(z) # Sigmoid function

        

        g = gradient(x, h, y) # Calculate the gradient

        w -= lr * g # Calculate the step size and perform the gradient descent with lr

        

        z = np.dot(x, w) # Update the parameters

        h = sigmoid(z) # Get Sigmoid value

        

        l = loss(h, y) # Get loss value

        

    return l, w # Return the gradient and parameters after iteration
"""

Set parameters and train

"""



x = df[['X0','X1']].values

y = df['Y'].values

lr = 0.001 # Learning rate

num_iter = 10000 # Iterations



#Train

L = Logistic_Regression(x, y, lr, num_iter)

L
"""

Plot the above results

"""



plt.figure(figsize=(10, 6))

plt.scatter(df['X0'],df['X1'], c=df['Y'])



x1_min, x1_max = df['X0'].min(), df['X0'].max(),

x2_min, x2_max = df['X1'].min(), df['X1'].max(),



xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

grid = np.c_[xx1.ravel(), xx2.ravel()]



probs = (np.dot(grid, np.array([L[1][1:3]]).T) + L[1][0]).reshape(xx1.shape)

plt.contour(xx1, xx2, probs, levels=[0], linewidths=1, colors='red');
"""

Draw the changing process of loss function

"""



def Logistic_Regression(x, y, lr, num_iter):

    intercept = np.ones((x.shape[0], 1)) # Initialize intercept as 1

    x = np.concatenate((intercept, x), axis=1)

    w = np.zeros(x.shape[1]) # Initialize parameters as 1

    

    l_list = [] # Save loss function value

    for i in range(num_iter): # Gradient descent iterations

        z = np.dot(x, w) # Linear function

        h = sigmoid(z) # Sigmoid function

        

        g = gradient(x, h, y) # Calculate the gradient

        w -= lr * g # Calculate the step size and perform the gradient descent with lr

        

        z = np.dot(x, w) # Update the parameters

        h = sigmoid(z) # Get Sigmoid value

        

        l = loss(h, y) # Get loss value

        l_list.append(l)

        

    return l_list



lr = 0.01 # Learning rate

num_iter = 30000 # Iterations

l_y = Logistic_Regression(x, y, lr, num_iter) # Train



#Plot

plt.figure(figsize=(10, 6))

plt.plot([i for i in range(len(l_y))], l_y)

plt.xlabel("Number of iterations")

plt.ylabel("Loss function")
from sklearn.linear_model import LogisticRegression



model = LogisticRegression(tol=0.001, max_iter=10000) # Set the same learning rate and iterations

model.fit(x, y)

model.coef_, model.intercept_
"""

Plot a graph

"""



plt.figure(figsize=(10, 6))

plt.scatter(df['X0'],df['X1'], c=df['Y'])



x1_min, x1_max = df['X0'].min(), df['X0'].max(),

x2_min, x2_max = df['X1'].min(), df['X1'].max(),



xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

grid = np.c_[xx1.ravel(), xx2.ravel()]



probs = (np.dot(grid, model.coef_.T) + model.intercept_).reshape(xx1.shape)

plt.contour(xx1, xx2, probs, levels=[0], linewidths=1, colors='red');
model.score(x, y)