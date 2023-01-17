import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/Housing Price data set.csv')

df_top = df.head()

df_tops = df_top.columns

#print(df_tops)

x_features = df_tops[2:]

#x_features

x_df = df[x_features]

y_df = df['price']
x_df = x_df.replace({'yes': 1, 'no': 0})

print(x_df.min())

print()

print(x_df.max())
x_df = (x_df - np.mean(x_df))/(x_df.max()-x_df.min())

x_df.insert( 0 ,"b",1)

x_df
m = x_df.shape[0] # Training Examples 

n = x_df.shape[1] # Number of features
lambdaa = 50

I = np.identity(n)

I[0][0] = 0
I


def NormalEquationRegularization(x_df,y_df):

    "Linear Regression using Normal Equation"

    transX = x_df.transpose()

    first = np.dot(transX,x_df)

    second = lambdaa*I

    first = first + second

    try:

        first = np.linalg.inv(first)

    except np.linalg.LinAlgError:

        print("X is not invertible")

    second = np.dot(transX,y_df)

    second = second.reshape(second.size, 1)

    return np.dot(first,second)
weights = NormalEquationRegularization(x_df,y_df)

weights
x_df 
y_df = y_df.values.reshape(y_df.size,1)

y_df.shape
predictions = np.dot(x_df,weights)
simple_error = y_df - predictions

print(simple_error)
squared_error = simple_error**2

print(squared_error)
squared_error_value = sum(squared_error)/len(squared_error)

root_mean_squared_error = squared_error_value ** (1/2)
print('root_mean_squared_error' + str(root_mean_squared_error))