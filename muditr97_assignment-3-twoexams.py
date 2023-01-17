import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/TwoExams.csv',header = None, names = ["exam1","exam2","result"])
df['exam1']
x_df = df[["exam1","exam2"]]

y_df = df['result']

X_train, X_test, Y_train, Y_test = tts(x_df, y_df, test_size = 0.3, random_state = 5)
X_train = (X_train - X_train.min())/(X_train.max()-X_train.min())

X_test = (X_test - X_test.min())/(X_test.max()-X_test.min())

X_train.insert(0, "b", 1)

X_test.insert(0, "b", 1)
X_train = np.array(X_train)

weights = np.zeros(len(X_train[0]))

Y_train = Y_train.values.reshape(Y_train.size, 1)
def sigmoid(z):

    return 1 / (1 + np.exp(-z))



def loss(h, y):

    return (-y * np.log(0.00000001+h) - (1 - y) * np.log(0.00000001+1 - h)).mean()
iterations = 1200

alpha = 0.005

cost = []
m = len(weights)

for i in range(iterations):

    z = X_train.dot(weights)

    hypothesis = sigmoid(z) 

    hypothesis = np.reshape(hypothesis, (hypothesis.size, 1) )

    hypothesis = hypothesis-Y_train

    X_trans = X_train.transpose()

    hypothesis = X_trans.dot(hypothesis)

    weights = np.reshape(weights, (weights.size, 1))

    weights = weights - (alpha/m) * hypothesis

    cost.append(loss(sigmoid(X_train.dot(weights)), Y_train))
prediction = sigmoid(X_test.dot(weights))

prediction.loc[prediction[0] < 0.5 , 0] = 0

prediction.loc[prediction[0] >= 0.5 , 0] = 1


plt.plot(np.arange(1, iterations+1), cost)

plt.xlabel("Number of Iteartions")

plt.ylabel("Cost")

plt.title("Gradient Descent Algorithm")

plt.show()
simple_error = Y_test - prediction

c = 0



for i in range(Y_test.shape[0]):

   # print(i)

   # print(pred.iloc[i][0])

   # print(float(Y_test.iloc[i]))

    if(prediction.iloc[i][0]==Y_test.iloc[i]):

        c += 1

    

accuracy = 100*c/len(Y_test)

print("accuracy : " + str(accuracy))
prediction = sigmoid(X_train.dot(weights))

simple_error = Y_train-prediction

print(simple_error)
squared_error = simple_error**2

print(squared_error)
squared_error_value = sum(squared_error)/len(squared_error)
print("squared_error_value = " + str(squared_error_value))