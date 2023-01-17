import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
def computeCost(X,y,thetas):

    m = X.shape[0]

    h = X.dot(thetas)

    J = (h-y).T.dot((h-y))/(2*m)

    return(J)
def computeGradientDescendWithCost(X, y, alpha, iterations):

    m = X.shape[0]

    n = X.shape[1]

    thetas = np.zeros(X.shape[1])

    vCosts = np.zeros(iterations)

    vThetas = np.zeros((iterations, n))

    for it in range(iterations):

        h = X.dot(thetas)

        thetas -= (alpha/m)*X.T.dot(h-y)

        vCosts[it] = computeCost(X,y,thetas)

        vThetas[it] = thetas

    return(vCosts, vThetas, thetas)
df = pd.read_csv("../../kaggle/input/housingprices/ex1data1.txt", header=None, names=['population','profit'])

df
fig = {

    'data': [{

        'type': 'scatter',

        'x': df['population'],

        'y': df['profit'],

        'mode': 'markers'

    }],

    'layout':{

        'title': 'LinearRegression',

        'xaxis': {'title': 'Population'},

        'yaxis': {'title': 'Profit'},

    }

}

iplot(fig)
X = np.array([np.ones(df.shape[0]), df['population']]).T

y = np.array(df['profit'])

alpha = 0.01

iterations = 2000

print("Number of rows: {}". format(X.shape[0]))

print("Number of theta parameters: {}". format(X.shape[1]))

print("Learning rate used: {}".format(alpha))

print("Number of iterations: {}".format(iterations))
vCosts, vThetas, thetas = computeGradientDescendWithCost(X, y, alpha, iterations)

print("Cost at first iteration = {}".format(vCosts[0]))

print("Theta parameters at first iteration = {}".format(vThetas[0]))

print("Cost at last iteration = {}".format(vCosts[iterations-1]))

print("Theta parameters at last iteration = {}".format(vThetas[iterations-1]))
point = np.array([1, 16])

prediction = thetas.dot(point)

prediction
fig = {

    'data': 

    [{

        'type': 'scatter',

        'x': X[:,1],

        'y': y,

        'name': 'training set',

        'mode': 'markers'

    },

    {

        'type': 'scatter',

        'x': X[:,1],

        'y': X.dot(thetas),

        'name': 'hypothesis',

        'mode': 'lines'   

    },

    {

        'type': 'scatter',

        'x': [point[1]],

        'y': [prediction],

        'name': 'prediction',

        'mode': 'markers'   

    }

    ],

    'layout':

    {

        'title': 'LinearRegression',

        'xaxis': {'title': 'Population'},

        'yaxis': {'title': 'Profit'},

    }

}

iplot(fig)
fig = {

    'data': 

    [{

        'type': 'scatter',

        'x': np.arange(iterations),

        'y': vCosts,

        'name': 'training set',

        'mode': 'lines'

    }],

    'layout':

    {

        'title': 'LinearRegression',

        'xaxis': {'title': 'Iterations'},

        'yaxis': {'title': 'Cost'}

    }

}

iplot(fig)
delta = 0.1

th0 = np.arange(-10.0, 10.0, delta)

th1 = np.arange(-1.0, 4.0, delta)

theta0, theta1 = np.meshgrid(th0, th1)

costes = np.zeros(theta0.shape)



for i in range(theta0.shape[0]):

    for j in range(theta0.shape[1]):

        costes[i,j]=computeCost(X,y,[theta0[i,j],theta1[i][j]])



fig = {

    'data': 

    [{

        'type': 'contour',

        'z': costes,

        'x': th0,

        'y': th1,

        'contours':{'start':0, 'end':700, 'size':25, 'showlabels': True, 'labelfont':{'size':12, 'color':'white'}}

    }],

    'layout':

    {

        'title': 'LinearRegression',

        'xaxis': {'title': 'theta0'},

        'yaxis': {'title': 'theta1'}

    }

}

iplot(fig)
data = pd.read_csv('../../kaggle/input/housingprices/ex1data2.txt', header=None, names=['size', 'bedrooms', 'price'])

target = 'price'

variables = [var for var in data.columns if var != target]

data
vMu = np.array([np.mean(data[var]) for var in variables])

vSigma = np.array([np.std(data[var]) for var in variables])



df = pd.DataFrame()

i=0

for col in variables:

    df[col] = (data[col]-vMu[i])/vSigma[i]

    i += 1

df[target] = data[target]/1000
X = np.array([np.ones(df.shape[0]), df['size']]).T

y = np.array(df['price'])

print(X.shape[0])

print(X.shape[1])



alpha = 0.01

iterations = 2000



vCosts, vThetas, thetas = computeGradientDescendWithCost(X, y, alpha, iterations)

print(vCosts[0])

print(vThetas[0])

print(vCosts[iterations-1])

print(vThetas[iterations-1])
raw_point = np.array([1.0, 1750.0])

point = raw_point

point[1] = (point[1]-vMu[0])/vSigma[0]

prediction = thetas.dot(point)

print(point)

print(prediction)
fig = {

    'data': 

    [{

        'type': 'scatter',

        'x': X[:,1],

        'y': y,

        'name': 'training set',

        'mode': 'markers'

    },

    {

        'type': 'scatter',

        'x': X[:,1],

        'y': X.dot(thetas),

        'name': 'hypothesis',

        'mode': 'lines'   

    },

    {

        'type': 'scatter',

        'x': [point[1]],

        'y': [prediction],

        'name': 'prediction',

        'mode': 'markers'   

    }    ],

    'layout':

    {

        'title': 'LinearRegression',

        'xaxis': {'title': 'Size'},

        'yaxis': {'title': 'Price'},

    }

}

iplot(fig)
fig = {

    'data': 

    [{

        'type': 'scatter',

        'x': np.arange(iterations),

        'y': vCosts,

        'name': 'training set',

        'mode': 'lines'

    }],

    'layout':

    {

        'title': 'LinearRegression',

        'xaxis': {'title': 'Iterations'},

        'yaxis': {'title': 'Cost'}

    }

}

iplot(fig)
delta = 1

th0 = np.arange(300.0, 400.0, delta)

th1 = np.arange(90.0, 120.0, delta)

theta0, theta1 = np.meshgrid(th0, th1)

costes = np.zeros(theta0.shape)



for i in range(theta0.shape[0]):

    for j in range(theta0.shape[1]):

        costes[i,j]=computeCost(X,y,[theta0[i,j],theta1[i][j]])



fig = {

    'data': 

    [{

        'type': 'contour',

        'z': costes,

        'x': th0,

        'y': th1,

        'contours':{'start':1000, 'end':7000, 'size':50, 'showlabels': True, 'labelfont':{'size':12, 'color':'white'}}

    }],

    'layout':

    {

        'title': 'LinearRegression',

        'xaxis': {'title': 'theta0'},

        'yaxis': {'title': 'theta1'}

    }

}

iplot(fig)
X = np.array([np.ones(df.shape[0]), df['size'], df['bedrooms']]).T

y = np.array(df['price'])

print(X.shape[0])

print(X.shape[1])



alpha = 0.01

iterations = 5000



vCosts, vThetas, thetas = computeGradientDescendWithCost(X, y, alpha, iterations)

print(vCosts[0])

print(vThetas[0])

print(vCosts[iterations-1])

print(vThetas[iterations-1])
from sklearn.linear_model import LinearRegression



X = df[['size','bedrooms']].values

y = df['price'].values



model_lr = LinearRegression()

model_lr.fit(X,y)



print("La ordenada al origen es: " + str(model_lr.intercept_))

print("Las coeficientes theta son: " + str(model_lr.coef_))



point = np.array([[1650, 3]])

prediction = model_lr.predict(point)

print("La prediccion del punto " + str(point)+ " es: " + str(prediction))