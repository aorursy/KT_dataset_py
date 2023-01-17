import warnings

warnings.filterwarnings("ignore")

from sklearn.datasets import load_boston

from random import seed

from random import randrange

from csv import reader

from math import sqrt

from sklearn import preprocessing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from prettytable import PrettyTable

from sklearn.linear_model import SGDRegressor

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error
boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

boston.head()
X = load_boston().data

Y = load_boston().target
import seaborn as sns

correlation_matrix = boston.corr().round(2)

# annot = True to print the values inside the square

sns.heatmap(data=correlation_matrix, annot=True)
scaler = preprocessing.StandardScaler().fit(X)

X = scaler.transform(X)
clf = SGDRegressor()

clf.fit(X, Y)

MSE_sgd = mean_squared_error(Y, clf.predict(X)) 

print(MSE_sgd)
def custom_SGD(W, X, Y, learning_rate=0.0001, max_iter=1000, sample_size=500):

    # W := Initial weight vector

    # X := Input Variable Vector

    # Y := Actual Y 

    # learning_rate := learning rate for SGD calculation, default value is 0.0001

    # max_iter := Maximum iteration for SGD calculation, default value is 1000

    # sample_size := Sample size for SGD, default value is 500

    X = np.append(np.ones((X.shape[0], 1)), X, axis=1) #adding column of ones to the left for easy dot product computation

    W = np.append(np.zeros((1,1)), W, axis=0) #adding 0 as the first element of W for easy dot product computation

    

    if sample_size > X.shape[0]:

        return "Invalid Sample Size." # If sample size is greater than number of datapoints, then return error

   

    for i in range(max_iter):

        idx = np.random.choice(np.arange(len(X)), sample_size, replace=False) # getting random indexes

        X_sample = X[idx] # sampling idx indexes from X

        Y_sample = Y[idx] # sampling idx indexes from Y

        W = W + learning_rate * np.dot(X_sample.T, (Y_sample - (np.dot(X_sample, W)))) # Updating weight vector

        

    return W[1:], W[0] # W[1:] is weight vector, W[0] is bias term
W = np.zeros((X.shape[1], 1))

W, B = custom_SGD(W,X, Y.reshape(-1,1), 0.0001, 1000, 300)
Y_pred = np.dot(X, W) + B

MSE_custom = mean_squared_error(Y, Y_pred)

print("MSE of custom implementation: {}".format(MSE_custom))
fig,ax = plt.subplots(figsize=(20, 10))



ax.scatter(Y, Y_pred)

ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=3, color='orange')

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')

ax.set_title('Actual Vs Predicted Values Plot (Using Custom Implementation)')

fig.show()
W_sgd = clf.coef_.tolist()

W_custom = W.flatten().tolist()



from prettytable import PrettyTable

x = PrettyTable()

x.field_names = ["SGD Weights", "Custom Weights"]

for i in range(len(W_sgd)):

    x.add_row([W_sgd[i], W_custom[i]])

print(x)
print('SGD Mean Squared Error =', MSE_sgd)

print('Custom Mean Squared Error =', MSE_custom)