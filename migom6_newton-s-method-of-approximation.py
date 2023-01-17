import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
missing_values = ["n/a", "na", "--"]

exam_df = pd.read_csv('../input/exam.csv', header = None, na_values = missing_values)

mc_df = pd.read_csv('../input/microchip.csv', header = None, na_values = missing_values)
exam_df.info()
exam_df.head()
x_features = [0,1]

y_features = [2]

x_exam = exam_df[x_features]

y_exam = exam_df[y_features]
x_exam.head()
y_exam.head()
# sns.catplot(x="x1", y="x2", data=);

sns.scatterplot(x=0,y=1,hue=2,data=exam_df)

# plt.plot([0, 4], [1.5, 0], linewidth=2)
X_train, X_test, Y_train, Y_test = tts(x_exam, y_exam, test_size = 0.3, random_state = 5)
X_train = (X_train - np.mean(X_train))/np.std(X_train)

X_test = (X_test - np.mean(X_test))/np.std(X_test)
X_train = X_train.assign(b=1)

X_test = X_test.assign(b=1)
X_train.head()
m = len(Y_train)

n = len(X_train.columns)

weights = np.zeros((n,1))

weights
def sigmoid(value):

    return 1/(1+np.exp(-value))
def GD_predict(X, weights):

    p = np.dot(X, weights)

    if(p >= 0.5):

        return 1

    else:

        return 0
def newtons(X, y, weights,iterations):

    m = len(y)

    n = len(weights)

        

    for _ in range(iterations):

        hessian = np.zeros((n,n))

        for j in range(n):

            for k in range(n):

                hessian[j, k] = sum(

                    X.iloc[i][j] * X.iloc[i][k] *

                    sigmoid(np.dot(X.iloc[i], weights)) *

                    (1 - sigmoid(np.dot(X.iloc[i], weights)))

                    for i in range(m)

                )

                    

        hypothesis = sigmoid(np.dot(X, weights))

        temp =  hypothesis - y

        jacobian = np.dot(X.T, temp)

        k =  np.dot(np.linalg.inv(hessian),jacobian) 

        weights = weights - (1/m*m)*k

    return weights
num_iters = 5
newtons_weights = newtons(X_train, Y_train, weights, num_iters)
GD_predict(X_test.iloc[2], newtons_weights)
Y_test.iloc[2]
Y_predicted = [GD_predict(x, newtons_weights) for x in X_test.values]

cm = confusion_matrix(Y_test, Y_predicted)

print("Confusion Matrix",cm)

ax = sns.heatmap(confusion_matrix(Y_test, Y_predicted))
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

recall = (cm[1][1])/(cm[1][1] + cm[0][1])

precision = (cm[1][1])/(cm[1][1] + cm[0][0])

print(accuracy)
mc_df.info()
mc_df.head()
x_features = [0,1]

y_features = [2]

x_mc = mc_df[x_features]

y_mc = mc_df[y_features]
x_mc.head()
y_mc.head()
sns.scatterplot(x=0,y=1,hue=2,data=mc_df)

x_mc[2] = np.square(x_mc[0])

x_mc[3] = np.square(x_mc[1])
x_mc.head()
X_train, X_test, Y_train, Y_test = tts(x_mc, y_mc, test_size = 0.3, random_state = 5)
X_train = (X_train - np.mean(X_train))/np.std(X_train)

X_test = (X_test - np.mean(X_test))/np.std(X_test)
X_train = X_train.assign(b=1)

X_test = X_test.assign(b=1)
X_train.head()
m = len(Y_train)

n = len(X_train.columns)

weights = np.zeros((n,1))

weights
alpha = 0.01

num_iters = 5
newtons_weights = newtons(X_train, Y_train, weights, num_iters)
Y_predicted = [GD_predict(x, newtons_weights) for x in X_test.values]

cm = confusion_matrix(Y_test, Y_predicted)

print("Confusion Matrix",cm)

ax = sns.heatmap(confusion_matrix(Y_test, Y_predicted))
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

recall = (cm[1][1])/(cm[1][1] + cm[0][1])

precision = (cm[1][1])/(cm[1][1] + cm[0][0])

print(accuracy)