import numpy as np

import pandas as pd

import seaborn as sns

import plotly

%matplotlib inline

import plotly.plotly as py

import matplotlib.pyplot as plt

from matplotlib import style
df = pd.read_csv("../input/data.csv")

df.head()
import seaborn as sns

fig = plt.subplots(figsize = (10,10))

sns.set(font_scale=1.5)

sns.heatmap(df.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})

plt.show()
def predictPrice(x,theta):

    return np.dot(x,theta)



def calculateCost(x,theta,Y):

    prediction = predictPrice(x,theta)

    return ((prediction - Y)**2).mean()/2
def abline(x,theta,Y):

    """Plot a line from slope and intercept"""

    

    y_vals = predictPrice(x,theta)

    plt.xlim(0, 20)

    plt.ylim(-10, 60)

    plt.xlabel('No. of Rooms in the house')

    plt.ylabel('Price of house')

    plt.gca().set_aspect(0.1, adjustable='datalim')

    plt.plot(x,Y,'.',x, y_vals, '-')

    plt.show()



def gradientDescentLinearRegression(alpha=0.047,iter=5000):

    theta0 = []

    theta1 = []

    costs = []

    predictor = df["floors"]

    x = np.column_stack((np.ones(len(predictor)),predictor))

    Y = df["price"]

    theta = np.zeros(2)

    for i in range(iter):

        pred = predictPrice(x,theta)

        t0 = theta[0] - alpha *(pred - Y).mean()

        t1 = theta[1] - alpha *((pred - Y)* x[:,1]).mean()

        

        theta = np.array([t0,t1])

        J = calculateCost(x,theta,Y)

        theta0.append(t0)

        theta1.append(t1)

        costs.append(J)

        if i%1000==0:

            print(f"Iteration: {i+1},Cost = {J},theta = {theta}")

            abline(x,theta,Y)

    print(f'theta0 = {len(theta0)}\ntheta1 = {len(theta1)}\nCosts = {len(costs)}')

    

gradientDescentLinearRegression()