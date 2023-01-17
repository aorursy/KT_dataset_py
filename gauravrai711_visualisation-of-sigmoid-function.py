# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Plotting Sigmoid Function



import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import matplotlib.colors

import pandas as pd





def sigmoid(x, w, b):

  return 1/(1 + np.exp(-(w*x + b)))



sigmoid(1, 0.5, 0)



w = -1.8    #@param {type: "slider", min: -2, max: 2, step: 0.1}

b = -0.5  #@param {type: "slider", min: -2, max: 2, step: 0.1}

X = np.linspace(-10,10,100)

Y = sigmoid(X, w, b)



plt.plot(X, Y)

plt.show()





def sigmoid_2d(x1, x2, w1, w2, b):

  return 1/(1 + np.exp(-(w1*x1 + w2*x2 + b)))



sigmoid_2d(1, 0, 0.5, 0, 0)



X1 = np.linspace(-10, 10, 100)

X2 = np.linspace(-10, 10, 100)



XX1, XX2 = np.meshgrid(X1, X2)



print(X1.shape, X2.shape, XX1.shape, XX2.shape)



w1 = 2

w2 = -0.5

b = 0

Y = sigmoid_2d(XX1, XX2, w1, w2, b)



my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","green"])



plt.contourf(XX1, XX2, Y, cmap = my_cmap, alpha = 0.6)

plt.show()



fig = plt.figure()

ax = plt.axes(projection='3d')

ax.plot_surface(XX1, XX2, Y, cmap='viridis')

ax.set_xlabel('x1')

ax.set_ylabel('x2')

ax.set_zlabel('y')



ax.view_init(30, 270)





# Compute Loss for a Given Dataset



w_unknown = 0.5

b_unknown = 0.25



X = np.random.random(25) * 20 - 10

Y = sigmoid(X, w_unknown, b_unknown)



plt.plot(X, Y, '*')

plt.show()



def calculate_loss(X, Y, w_est, b_est):

  loss = 0

  for x, y in zip(X, Y):

    loss += (y - sigmoid(x, w_est, b_est))**2

  return loss



W = np.linspace(0, 2, 101)

B = np.linspace(-1, 1, 101)



WW, BB = np.meshgrid(W, B)



Loss = np.zeros(WW.shape)



WW.shape



for i in range(WW.shape[0]):

  for j in range(WW.shape[1]):

    Loss[i, j] = calculate_loss(X, Y, WW[i, j], BB[i, j])



fig = plt.figure()

ax = plt.axes(projection='3d')

ax.plot_surface(WW, BB, Loss, cmap='viridis')

ax.set_xlabel('w')

ax.set_ylabel('b')

ax.set_zlabel('Loss')



ax.view_init(30, 270)



ij = np.argmin(Loss)

i = int(np.floor(ij/Loss.shape[1]))

j = int(ij - i * Loss.shape[1])



print(i, j)



print(WW[i, j], BB[i, j])



# Class for Sigmoid Neuron



class SigmoidNeuron:

  

  def __init__(self):

    self.w = None

    self.b = None

    

  def perceptron(self, x):

    return np.dot(x, self.w.T) + self.b

  

  def sigmoid(self, x):

    return 1.0/(1.0 + np.exp(-x))

  

  def grad_w(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    return (y_pred - y) * y_pred * (1 - y_pred) * x

  

  def grad_b(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    return (y_pred - y) * y_pred * (1 - y_pred)

  

  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True):

    

    # initialise w, b

    if initialise:

      self.w = np.random.randn(1, X.shape[1])

      self.b = 0

    

    for i in range(epochs):

      dw = 0

      db = 0

      for x, y in zip(X, Y):

        dw += self.grad_w(x, y)

        db += self.grad_b(x, y)       

      self.w -= learning_rate * dw

      self.b -= learning_rate * db



# Fit for toy data



X = np.asarray([[2.5, 2.5], [4, -1], [1, -4], [-3, 1.25], [-2, -4], [1, 5]])

Y = [1, 1, 1, 0, 0, 0]



sn = SigmoidNeuron()

sn.fit(X, Y, 1, 0.25, True)



def plot_sn(X, Y, sn, ax):

  X1 = np.linspace(-10, 10, 100)

  X2 = np.linspace(-10, 10, 100)

  XX1, XX2 = np.meshgrid(X1, X2)

  YY = np.zeros(XX1.shape)

  for i in range(X2.size):

    for j in range(X1.size):

      val = np.asarray([X1[j], X2[i]])

      YY[i, j] = sn.sigmoid(sn.perceptron(val))

  ax.contourf(XX1, XX2, YY, cmap=my_cmap, alpha=0.6)

  ax.scatter(X[:,0], X[:,1],c=Y, cmap=my_cmap)

  ax.plot()



sn.fit(X, Y, 1, 0.05, True)

N = 30

plt.figure(figsize=(10, N*5))

for i in range(N):

  print(sn.w, sn.b)

  ax = plt.subplot(N, 1, i + 1)

  plot_sn(X, Y, sn, ax)

  sn.fit(X, Y, 1, 0.5, False)