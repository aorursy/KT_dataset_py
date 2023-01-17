from sklearn.datasets import load_iris

import pandas as pd
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
iris = load_iris()

X = pd.DataFrame(data=iris.data, columns=iris.feature_names)

y = pd.DataFrame(data=iris.target, columns=['types'])

data = pd.concat([X, y], axis=1)

data.head(3)
plt.scatter(data['petal length (cm)'], data['petal width (cm)'], c=data.types, cmap='autumn', s=60)

plt.show()
data = data[data['types'] != 0]

plt.scatter(data['petal length (cm)'], data['petal width (cm)'], c=data.types, cmap='autumn', s=60)
from sklearn.linear_model import LogisticRegression
col = ['x1', 'x2', 'x3', 'x4', 'y'] 

data.columns = col

data['y'] = data.y.replace({1: 0, 2: 1})

data.head(3)
x1, x2, x3, x4, y = data.x1, data.x2, data.x3, data.x4, data.y

EPOCHS = 50

lr =  0.01



costs = []

preds = []

params = np.random.normal(size=(5,))



for _ in range(EPOCHS):

    predictions = params[0] + params[1] * x1 + params[2] * x2 + params[3] * x3 + params[4] * x4

    

    h = 1. / (1 + np.exp(-predictions))

    cost = np.sum(((-y) * np.log(h)) - ((1 - y) * np.log(1 - h))) / len(predictions)

    costs.append(cost)



    params[0] -= lr * np.sum(h - y) / len(predictions)

    params[1] -= lr * np.sum(x1 * (h - y))/ len(predictions)

    params[2] -= lr * np.sum(x2 * (h - y)) / len(predictions)

    params[3] -= lr * np.sum(x3 * (h - y)) / len(predictions)

    params[4] -= lr * np.sum(x4 * (h - y)) / len(predictions)

    

    if _ % 5 == 0:

        print (cost)

    

print (params[0], params[1], params[2], params[3], params[4])
plt.plot(costs)
""" Decision is mine """

x1, x2, x3, x4, y = data.x1, data.x2, data.x3, data.x4, data.y

EPOCHS = 100

lr =  0.5

alpha = 0.8

exp_avg_0, exp_avg_1, exp_avg_2, exp_avg_3, exp_avg_4 = 0, 0, 0, 0, 0



costs = []

preds = []

params = np.random.normal(size=(5,))



for _ in range(EPOCHS):

    nester_0 = params[0] - (alpha * exp_avg_0)

    nester_1 = params[1] - (alpha * exp_avg_1)

    nester_2 = params[2] - (alpha * exp_avg_2)

    nester_3 = params[3] - (alpha * exp_avg_3)

    nester_4 = params[4] - (alpha * exp_avg_4)

    predictions = nester_0 + nester_1 * x1 + nester_2 * x2 + nester_3 * x3 + nester_4 * x4

    

    h = 1. / (1 + np.exp(-predictions))

    cost = np.sum(((-y) * np.log(h)) - ((1 - y) * np.log(1 - h))) / len(predictions)

    costs.append(cost)



    exp_avg_0 = (alpha * exp_avg_0) + (lr*(1-alpha)) * np.sum(h - y) / len(predictions)

    exp_avg_1 = (alpha * exp_avg_1) + (lr*(1-alpha)) * np.sum(x1 * (h - y)) / len(predictions)

    exp_avg_2 = (alpha * exp_avg_2) + (lr*(1-alpha)) * np.sum(x2 * (h - y)) / len(predictions)

    exp_avg_3 = (alpha * exp_avg_3) + (lr*(1-alpha)) * np.sum(x3 * (h - y)) / len(predictions)

    exp_avg_4 = (alpha * exp_avg_4) + (lr*(1-alpha)) * np.sum(x4 * (h - y)) / len(predictions)

    

    

    params[0] -= exp_avg_0

    params[1] -= exp_avg_1

    params[2] -= exp_avg_2

    params[3] -= exp_avg_3

    params[4] -= exp_avg_4

    

    if _ % 10 == 0:

        print (cost)

    

print (params[0], params[1], params[2], params[3], params[4])
plt.plot(costs)
""" Decision is mine """

x1, x2, x3, x4, y = data.x1, data.x2, data.x3, data.x4, data.y

EPOCHS = 100

lr =  0.05

alpha = 0.7

exp_avg_0, exp_avg_1, exp_avg_2, exp_avg_3, exp_avg_4 = 0, 0, 0, 0, 0

eps = 1



costs = []

preds = []

params = np.random.normal(size=(5,))



for _ in range(EPOCHS):

    

    predictions = params[0] + params[1] * x1 + params[2] * x2 + params[3] * x3 + params[4] * x4

    

    h = 1. / (1 + np.exp(-predictions))

    cost = np.sum(((-y) * np.log(h)) - ((1 - y) * np.log(1 - h))) / len(predictions)

    costs.append(cost)



    exp_avg_0 = (alpha * exp_avg_0) + (lr*(1-alpha)) * ((np.sum(h - y) / len(predictions))**2)

    exp_avg_1 = (alpha * exp_avg_1) + (lr*(1-alpha)) * ((np.sum(x1 * (h - y)) / len(predictions))**2)

    exp_avg_2 = (alpha * exp_avg_2) + (lr*(1-alpha)) * ((np.sum(x2 * (h - y)) / len(predictions))**2)

    exp_avg_3 = (alpha * exp_avg_3) + (lr*(1-alpha)) * ((np.sum(x3 * (h - y)) / len(predictions))**2)

    exp_avg_4 = (alpha * exp_avg_4) + (lr*(1-alpha)) * ((np.sum(x4 * (h - y)) / len(predictions))**2)

    

    

    params[0] -= (lr / np.sqrt(exp_avg_0 + eps)) * (np.sum(h - y) / len(predictions))

    params[1] -= (lr / np.sqrt(exp_avg_1 + eps)) * (np.sum(x1 * (h - y)) / len(predictions))

    params[2] -= (lr / np.sqrt(exp_avg_2 + eps)) * (np.sum(x2 * (h - y)) / len(predictions))

    params[3] -= (lr / np.sqrt(exp_avg_3 + eps)) * (np.sum(x3 * (h - y)) / len(predictions))

    params[4] -= (lr / np.sqrt(exp_avg_4 + eps)) * (np.sum(x4 * (h - y)) / len(predictions))

    

    if _ % 10 == 0:

        print (cost)

    

print (params[0], params[1], params[2], params[3], params[4])
plt.plot(costs)