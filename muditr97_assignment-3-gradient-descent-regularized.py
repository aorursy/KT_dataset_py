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
x_df = (x_df - np.mean(x_df))/(x_df.max()-x_df.min())

x_df.insert( 0 ,"b",1)

x_df
m = x_df.shape[0] # Training Examples 

n = x_df.shape[1] # Number of features



x_df = np.array(x_df)

weights = np.zeros(len(x_df[0]))
y_df = y_df.values.reshape(y_df.size,1)

y_df.shape
iterations = 800

alpha = 0.001

cost = []

lambdaa = 50

lambdaa_factor = 1 - ((alpha*lambdaa)/m)
for i in range(iterations):

    hypothesis = np.dot(x_df,weights)

    hypothesis = np.reshape(hypothesis, (hypothesis.size, 1))

    hypothesis = hypothesis - y_df

    X_trans = x_df.transpose()

    hypothesis = X_trans.dot(hypothesis)

    weights = np.reshape(weights, (weights.size, 1))

    

    weights = lambdaa_factor*weights - (alpha/len(weights)) * hypothesis

    

    cost.append(float((sum(x_df.dot(weights) - y_df)**2))/2*len(weights))



plt.plot(np.arange(1, iterations+1), cost)

plt.xlabel("Iteartions")

plt.ylabel("Cost")

plt.title("Gradient Descent")

plt.show()



pred = x_df.dot(weights)



simple_error = y_df-pred

#print(simple_error)

squared_error = simple_error**2

#print(squared_error)

squared_error_value = sum(squared_error)/len(squared_error)

root_mean_squared_error = squared_error_value ** (1/2)

print("root_mean_squared_error_value = " + str(root_mean_squared_error))