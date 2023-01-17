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
X_train, X_test, Y_train, Y_test = tts(x_exam, y_exam, test_size = 0.7, shuffle = True)
X_test = (X_test - np.mean(X_train))/np.std(X_train)

X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_train = X_train.assign(b=1)

X_test = X_test.assign(b=1)
X_train.head()
m = len(Y_train)

n = len(X_train.columns)

weights = np.random.rand(X_train.shape[1], 1)

weights
def sigmoid(value):

    return 1/(1+np.exp(-value))
def GD_predict(X, weights):

    p = np.dot(X, weights)

    if(sigmoid(p) >= 0.5):

        return 1

    else:

        return 0

    

def predict(features, weights):

  z = np.dot(features, weights)

  return sigmoid(z)
def cost_fn(features, labels, weights):

  m = len(labels)

  predictions = predict(features, weights)

  cost = -labels*np.log(predictions) - (1-labels)*np.log(1-predictions)

  return cost.sum() / m



def ridgegradientDescentMulti(X, y, weights, alpha, iterations, lamda):

    cost_history = []

    m = len(y)

    for _ in range(iterations):

        temp = sigmoid(np.dot(X, weights)) - y

        temp = np.dot(X.T, temp)

        weights = weights - ((alpha/m) * temp)

        cost = cost_fn(X.values, y.values, weights)

        cost_history.append(cost)

    return weights,cost_history
alpha = 0.05

num_iters = 1000

lamda = 0.05
ridge_weights,cost_history = ridgegradientDescentMulti(X_train, Y_train, weights, alpha, num_iters, lamda)
GD_predict(X_test.iloc[2], ridge_weights)
Y_test.iloc[2]
Y_predicted = [GD_predict(x, ridge_weights) for x in X_test.values]

cm = confusion_matrix(Y_test, Y_predicted)

print("Confusion Matrix",cm)

ax = sns.heatmap(confusion_matrix(Y_test, Y_predicted))
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

recall = (cm[1][1])/(cm[1][1] + cm[0][1])

precision = (cm[1][1])/(cm[1][1] + cm[0][0])

print(accuracy)

# print(cost_history)

plt.plot(cost_history)
mc_df.head()
x_features = [0,1]

y_features = [2]

x_exam = np.square(mc_df[x_features])

y_exam = mc_df[y_features]
x_exam.head()
y_exam.head()
sns.scatterplot(x=0,y=1,hue=2,data=mc_df)
X_train, X_test, Y_train, Y_test = tts(x_exam, y_exam, test_size = 0.3, random_state = 5)
X_test = (X_test - np.mean(X_train))/np.std(X_train)

X_train = (X_train - np.mean(X_train))/np.std(X_train)
X_train = X_train.assign(b=1)

X_test = X_test.assign(b=1)

# X_train = X_train.values

# X_test = X_test.values
m = len(Y_train)

n = len(X_train.columns)

weights = np.random.rand(X_train.shape[1], 1)

weights
alpha = 0.01

num_iters = 1000

lamda = 0.05
ridge_weights,cost_history = ridgegradientDescentMulti(X_train, Y_train, weights, alpha, num_iters, lamda)
GD_predict(X_test.iloc[2], ridge_weights)
Y_predicted = [GD_predict(x, ridge_weights) for x in X_test.values]

cm = confusion_matrix(Y_test, Y_predicted)

print("Confusion Matrix",cm)

ax = sns.heatmap(confusion_matrix(Y_test, Y_predicted))
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

recall = (cm[1][1])/(cm[1][1] + cm[0][1])

precision = (cm[1][1])/(cm[1][1] + cm[0][0])

print(accuracy)

# print(cost_history)

plt.plot(cost_history)