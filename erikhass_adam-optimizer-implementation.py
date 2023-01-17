# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sys



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/electric-motor-temperature/pmsm_temperature_data.csv')
data.head()
data.columns
data.shape
target = pd.DataFrame(data, columns= ["pm"])

data = data.drop(["profile_id", "stator_yoke", "stator_winding", "stator_tooth", "pm"],1)
data.isna().any()
target.isna().any()
training_split = 0.8

m = data.shape[0]

training_size = int(training_split*m)



training_data = data.iloc[:training_size, :]

training_data.insert(0, "Bias", 1)

training_label = target.iloc[:training_size]



test_data = data.iloc[:training_size, :]

test_data.insert(0,"Bias",1)

test_label = target.iloc[:training_size]
def computeCost(data, target, theta, lamb = 0):

    m = data.shape[0]

    J = 2/m * np.sum(np.subtract(np.dot(data, theta), target)**2) + lamb * np.linalg.norm(theta)

    return J



def calculateGradient(data, target, theta, optParams):

    m = optParams["m"]

    lambd = optParams["lambd"]

    theta_grad = np.zeros(theta.shape)

    #don't apply the regularization to b (theta[0])

    theta_grad = 1/m * np.transpose(np.dot(np.transpose(np.subtract(np.dot(data, theta), target)),data)) + np.r_[np.zeros((1,1)), np.multiply(lambd/m, theta[1:,:])]

    return theta_grad

    

def calcBatch(data, target, theta, batchStart, batch_size, optParams):

    theta_grad = calculateGradient(data.iloc[batchStart:batchStart+batch_size,:], target.iloc[batchStart:batchStart+batch_size], theta, optParams)

    theta = updateTheta(theta, theta_grad, optParams)

    return theta

    

def updateTheta(theta, theta_grad, optParams):

    if optParams["optimizer"] == "sgd":

        learning_rate = optParams["lr"]

        theta = theta - learning_rate * theta_grad

    elif optParams["optimizer"] == "adam":

        beta1 = optParams["beta1"]

        beta2 = optParams["beta2"]

        epsilon = optParams["epsilon"]

        mt = optParams["mt"]

        vt = optParams["vt"]

        learning_rate = optParams["lr"]

        t = optParams["t"]

        m = beta1 * mt + (1.0 - beta1) * theta_grad

        v = beta2 * vt + (1.0 - beta2) * theta_grad**2

        #m_hat = np.divide(m, 1.0 - np.power(beta1, t))

        #v_hat = np.divide(m, 1.0 - np.power(beta2, t))

        lrt = learning_rate * np.sqrt(1.0 - beta2**t)/(1.0 - beta1**t)

        theta = theta - lrt * np.divide(m, (np.sqrt(v) + epsilon))

    else:

        theta = 0

        

    optParams["mt"] = m

    optParams["vt"] = v

    return theta
def train_model(data, target, epochs, batch_size, learning_rate = 0.01, lambd = 0, beta1 = 0.9, beta2=0.999, epsilon = 1e-8, optimizer="sgd"):



    theta = np.random.randn(data.shape[1],1)

    J_hist = np.zeros((epochs, 1))

    m = data.shape[0]

    numBatches = int(m/batch_size)

    calcLastBatch = m%batch_size > 0

    

    optParams = {}

    optParams["m"] = m

    optParams["lr"] = learning_rate

    optParams["lambd"] = lambd

    optParams["beta1"] = beta1

    optParams["beta2"] = beta2

    optParams["epsilon"] = epsilon

    optParams["mt"] = np.zeros((data.shape[1],1))

    optParams["vt"] = np.zeros((data.shape[1],1))

    optParams["optimizer"] = optimizer

    

    for i in range(epochs):

        J_hist[i] = computeCost(data, target, theta, lambd)

        optParams["t"] = float(i) + 1

        for currBatch in range(0):

            theta = calcBatch(data, target, theta, currBatch*batch_size, batch_size, optParams)  

        if calcLastBatch:

            theta = calcBatch(data, target, theta, numBatches*batch_size, m - numBatches*batch_size, optParams)

    

    return theta, J_hist

theta, J_hist = train_model(training_data, training_label, epochs = 5000, batch_size = 512, learning_rate= 0.0001, lambd = 1, optimizer = "adam")
np.mean(np.subtract(np.dot(test_data, theta),test_label)**2)
plt.plot(J_hist)
plt.figure(figsize=(20,10))

plt.plot(np.dot(test_data,theta))

plt.plot(test_label)
print(theta)