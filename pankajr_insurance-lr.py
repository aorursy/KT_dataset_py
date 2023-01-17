# import all libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Load dataset

df = pd.read_csv("/kaggle/input/insurance/ins.csv", delimiter=",")
# Print all data

df
#First five records

df.head(5)
#Last five records

df.tail(5)
#Rows and columns

df.shape
# Calculate slope(m)

X = df["X"]

Y = df["Y"]

numer = 0

denom = 0

for i in range(len(X)):

    numer += (X[i] - np.mean(X)) * (Y[i] - np.mean(Y))

    denom += (X[i] - np.mean(X)) ** 2

m = numer / denom

c = np.mean(Y) - (m * np.mean(X))

m,c
# Calculating line values x and y

x = np.linspace(np.min(X), np.max(X), 1000)

y = c + m * x



# Ploting Line

plt.plot(x, y, color='#58b970', label='Regression Line')

# Ploting Scatter Points

plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
# Calculating Root Mean Squares Error

rmse = 0

for i in range(len(X)):

    y_pred = c + m * X[i]

    rmse += (Y[i] - y_pred) ** 2

rmse = np.sqrt(rmse/len(X))

rmse
# Calculating R2 Score

ss_tot = 0

ss_res = 0

for i in range(len(X)):

    y_pred = c + m * X[i]

    ss_tot += (Y[i] - np.mean(Y)) ** 2

    ss_res += (Y[i] - y_pred) ** 2

r2 = 1 - (ss_res/ss_tot)

r2