import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
df = pd.read_csv("../input/HR_comma_sep.csv")
depart = df.Department.unique()

for i in range(len(depart)):

  df.Department = df.Department.replace(depart[i],i+1)
sal = df.salary.unique()

for i in range(len(sal)):

  df.salary = df.salary.replace(sal[i],i+1)
Y = df['left']

X = df.drop(['left'],axis=1)

test = X

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size=0.30)
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
ones = np.ones((X_train.shape[0],1),dtype=int)

X_train = np.concatenate((ones,X_train),axis=1)
plt.figure(figsize=(8,5))

sb.heatmap(data=df.corr(),annot=True,robust=True,linewidths=.01)
def sigmoid(z):

  return 1/(1+np.exp(-z))
def cost_function(x,y,w):

  h = sigmoid(np.dot(x,w))

  pred = np.sum((y*np.log(h))+((1-y)*np.log(1-h)))

  return -(pred/len(y))
def gradient_decent(x,y,theta,lr,epochs):

  m = len(y)

  err = []

  for i in range(epochs):

    theta = gradient_step(x,y,theta,lr)

    err.append(cost_function(x,y,theta))

  return theta,err
def gradient_step(x,y,theta,lr):

  m = len(y)

  h = sigmoid(np.dot(x,theta))

  pred = h - y

  deriv = np.sum((pred[:,None]*x),axis=0)

  theta -= (lr*(1/m))*deriv

  return theta
def predict(x,theta):

  ret = list()

  for i in range(len(x)):

    bit = x[i]

    bit = scaler.transform([bit])

    bit = np.insert(bit,0,1)

    pred = sigmoid(np.sum(bit*theta))

    if(pred >= 0.5):

      ret.append(1)

    else:

      ret.append(0)

  return ret
# hyper parameters

theta = np.zeros(X_train.shape[1])

lr = 0.5

epochs = 10000

theta,err = gradient_decent(X_train,y_train,theta,lr,epochs)
plt.plot(err)

plt.grid()

plt.show()
predictions = predict(X_test,theta)

accuracy_score(y_test,predictions)