import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn import datasets

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
data = datasets.load_iris()

df = pd.DataFrame(data.data,columns=['sepalL','sepalW','petalL','petalW'])

df['type'] = data.target

# - Iris-Setosa



# - Iris-Versicolour

# - Iris-Virginica
sb.heatmap(df.corr(),annot=True)
X = np.array(df.drop(['type'],axis=1))

Y = np.array(df['type'])
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.30)
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
ones = np.ones((X_train.shape[0],1),dtype=int)

X_train = np.concatenate((ones,X_train),axis=1)
def gradient_decent(x,y,theta,alpha,epochs,cats):

  m = x.shape[0]

  n = x.shape[1]

  err = [[],[],[]]

  for i in range(len(cats)):

    newY = np.array(y,copy=True)

    newY[y_train == i] = 1

    newY[y_train != i] = 0

    for j in range(epochs):

      theta[i] = gradient_step(x,newY,theta[i],alpha,m)

      err[i].append(cost_function(x,newY,theta[i]))

  return err,theta
def gradient_step(x,y,w,lr,m):

  h = sigmoid(np.dot(x,w))

  pred = h - y

  deriv = np.sum((pred[:,None]*x),axis=0)

  w -= (lr*(1/m))*deriv

  return w
def sigmoid(z):

  return 1/(1+np.exp(-z))
def cost_function(x,y,w):

  h = sigmoid(np.dot(x,w))

  h = np.sum( (y*np.log(h)) + ((1-y)*np.log(1-h)) )

  return -(h / len(y))
# hyper  parameters

theta = np.zeros((3,X_train.shape[1]))

lr = 0.1

iters = 1000

err , theta = gradient_decent(X_train,y_train,theta,lr,iters,[0,1,2])
print(theta)

for i in err:

  plt.grid()

  plt.plot(i)

  plt.show()
def bulk_predict(x,w):

  predBit = np.zeros(len(x),dtype=int)

  predPr = np.zeros((len(x),len(w)))

  for i in range(len(x)):

    bit = x[i]

    bit = scaler.transform([bit])

    bit = np.insert(bit,0,1)

    for j in range(len(theta)):

      p = sigmoid(np.sum(bit*w[j]))

      predPr[i][j] = round(p, 2)*100

    predBit[i] = predPr[i].argmax()

  return predBit,predPr
predBit,predPr  = bulk_predict(X_test,theta)

# - Iris-Setosa

# - Iris-Versicolour

# - Iris-Virginica

predBit
accuracy_score(predBit,y_test) * 100