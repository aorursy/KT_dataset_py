import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/car-sales/Car_sales.csv')

df.head()
df.isnull().sum()
df.dropna(inplace = True)
df.head()
sns.heatmap(df.isnull())
df.describe()
df = df[['Engine_size','Horsepower','Fuel_efficiency','Price_in_thousands']]

print(df.head())
plt.scatter(df['Engine_size'],df['Price_in_thousands'])
plt.scatter(df['Horsepower'],df['Price_in_thousands'])
plt.scatter(df['Fuel_efficiency'],df['Price_in_thousands'])
features = df[['Engine_size','Horsepower','Fuel_efficiency']].to_numpy()

print(features)
for i in range(3):

    min = np.min(features[:,i])

    max = np.max(features[:,i])

    features[:,i] = np.min(features[:,i] - min)/(max-min)
target = df['Price_in_thousands'].to_numpy()

print(target)
weights = np.random.rand(3) #randomly generated weights

print(weights)
b = np.random.rand(1) #randomly generated bias

bias = np.array([b[0] for i in range(len(features))])

print(bias)
def linear(features,weights,bias):

    y_hat = weights.dot(features.transpose()) + np.array([b[0] for i in range(len(features))])

    return y_hat
y_hat = linear(features,weights,b)

print(y_hat)
def meansquare(y,y_hat):

    MSE = np.sum((y-y_hat) **2) / len(y)

    return MSE
error = meansquare(target,y_hat)

print(error)
def gradient(target,features,weights,bias):

    m =len(features)

    target_pred = linear(features,weights,bias)

    loss = target - target_pred

    grad_bias = np.array([-2/m * np.sum(loss)])

    grad_weights = np.ones(3)

    feature_0 = np.array([feature[0] for feature in features])

    grad_weights[0] = -2/m * np.sum(loss * feature_0)

    feature_1 = np.array([feature[1] for feature in features])

    grad_weights[1] = -2/m * np.sum(loss * feature_1)

    feature_2 = np.array([feature[1] for feature in features])

    grad_weights[2] = -2/m * np.sum(loss * feature_2)

    return grad_bias,grad_weights
def grad_desc(learning_rate,epochs,target,features,weights,bias):

    MSE_list = []

    for i in range(epochs):

        grad_bias,grad_weights = gradient(target,features,weights,bias)

        weights -= grad_weights * learning_rate

        bias -= grad_bias * learning_rate

        new_pred = linear(features,weights,bias)

        mse_new = meansquare(target,new_pred)

        MSE_list.append(mse_new)

    return_dict = {'weights': weights, 'bias': bias[0], 'MSE': mse_new, 'MSE_list': MSE_list}

    return return_dict
model = grad_desc(0.0001,2000,target,features,weights,bias)
print("Weights- {}\nBias- {}\nMSE- {}".format(model['weights'], model['bias'], model['MSE']))
def linearmodel(model,feature_list):

    price = np.sum(model['weights'] * feature_list) + model['bias']

    return price
target_price = 196

feature_list = [2.0,4,8.5]

predicted_price = linearmodel(model, feature_list)

print("Price in thousands:",predicted_price)