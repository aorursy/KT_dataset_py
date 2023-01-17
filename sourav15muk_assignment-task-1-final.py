# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the boston price dataset

from sklearn.datasets import load_boston

boston = load_boston()

# Let's make a pandas dataframe with the data

data = pd.DataFrame(boston.data, columns= boston.feature_names)

data['Target']= boston.target
# A look at the data

data.head(5)
#checking for any missing values 

data.isna().sum()
# Let's standardize our data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data = scaler.fit_transform(data)



X = data[:,:13]

y = data[:,13]
# Let's do simple regression first

from sklearn.linear_model import LinearRegression

reg1 = LinearRegression()

reg1.fit(X,y)
# Looking at the coefficients or weights from simple linear regression

reg1_weights = reg1.coef_

reg1_weights
#Now implementing stochastic gradient descent



# Assuming the loss function to be L, error e and weights w, we can define the loss function differential with respect to w as



def dl_dw(X,err,w):

    return -2*(X.T).dot(err)/len(X)



# Now for sgd, 



def sgd(learning_rate= 0.001, epochs= 150, batch= 5, decay= 0.9): #experimenting with learning_rate and epochs may give more better results

    number_of_epoch_runs = int(len(X)/batch)

    

    # starting initially with random weights and loss

    w = np.zeros((13, 1))

    parameters = []

    loss = np.zeros((epochs,1))

    

    for i in range(epochs):

        parameters.append(w)

        

        for j in range(number_of_epoch_runs):

            idx = np.random.choice(len(X),batch, replace = False)

            err = y[idx] - X[idx].dot(w)

            

            #then updating our parameters

            w = w - (learning_rate)* dl_dw(X[idx],err,w)

            

        loss[i] = np.sum(np.square(err))/len(err)

        # We also deacy the learning rate as it progresses

        learning_rate = learning_rate * decay

        w

    

    return parameters, loss, w



parameters, loss, w = sgd()
import matplotlib.pyplot as plt

plt.plot(loss)
# Getting the weights

reg2_weights = w[:,-1]
# plotting the comparison between two models

import seaborn as sns

sns.regplot(reg1_weights, reg2_weights)