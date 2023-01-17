import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time
X = np.arange(100, dtype=float)
y = np.concatenate((np.zeros(50, dtype=float), np.ones(50, dtype=float)), axis=0)
plt.scatter(X, y)
def mse(y, y_pred): ##mean squared error
  return np.sum((y - y_pred)**2)/y.shape[0]
epoch_loss = []

slope = 0.
bias = 0.
learning_rate = 1e-6 
n = X.shape[0]

for epoch in range(10): 
  y_pred = slope*X + bias
  loss = mse(y, y_pred)
  epoch_loss.append(loss)

  ######plotting#####
  display.display(plt.gcf())
  display.clear_output(wait=True)
  fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
  ax0.scatter(X, y)
  ax0.plot(X, y_pred, 'r')
  ax0.set_title('slope = {0:.5f}, bias = {1:.5f}'.format(slope, bias))
  ax1.set_title('mse = {0:.2f}'.format(loss))
  ax1.plot(epoch_loss)
  plt.show()
  time.sleep(1)
  ###################
  
  ###slope and bias derivatives with respect to mse###
  D_mse_wrt_slope = -np.sum(X * (y - y_pred)) 
  D_mse_wrt_bias = -np.sum(y - y_pred) 
  
  
  slope = (slope - learning_rate * D_mse_wrt_slope)
  bias = (bias - learning_rate * D_mse_wrt_bias)
X = np.concatenate((X, np.array([200])))
y = np.concatenate((y, np.array([1])))
epoch_loss = []

slope = 0.
bias = 0.
learning_rate = 1e-6 
n = X.shape[0]

for epoch in range(10):
  y_pred = slope*X + bias
  loss = mse(y, y_pred)
  epoch_loss.append(loss)

  ######plotting#####
  display.display(plt.gcf())
  display.clear_output(wait=True)
  fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
  ax0.scatter(X, y)
  ax0.plot(X, y_pred, 'r')
  ax0.set_title('slope = {0:.5f}, bias = {1:.5f}'.format(slope, bias))
  ax1.set_title('mse = {0:.2f}'.format(loss))
  ax1.plot(epoch_loss)
  plt.show()
  time.sleep(1)
  ###################
  
  
  ###slope and bias derivatives with respect to mse###
  D_mse_wrt_slope = -np.sum(X * (y - y_pred))
  D_mse_wrt_bias = -np.sum(y - y_pred)

  slope = (slope - learning_rate * D_mse_wrt_slope)
  bias = (bias - learning_rate * D_mse_wrt_bias)
def log_loss(y, y_pred): ##log loss error (binary cross entropy)
  return -np.sum((y*np.log(y_pred) + (1-y)*np.log(1-y_pred)))/y.shape[0]
epoch_loss = []

slope = 0.
bias = 0.
learning_rate = 1e-5
n = X.shape[0]

for epoch in range(700000+1):
  linear = slope*X + bias
  y_pred = 1/(1+np.exp(-linear)) ##logistic
  loss = log_loss(y, y_pred)
  epoch_loss.append(loss)


  if(epoch%50000 == 0):
    ######plotting#####
    display.display(plt.gcf())
    display.clear_output(wait=True)
    fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
    fig.suptitle('epoch = {0}'.format(epoch))
    ax0.scatter(X, y)
    ax0.plot(X, y_pred, 'r')
    ax0.set_title('slope = {0:.5f}, bias = {1:.5f}'.format(slope, bias))
    ax1.set_title('loss = {0:.2f}'.format(loss))
    ax1.plot(epoch_loss)
    plt.show()
    time.sleep(1)
    ###################
    
  ###slope and bias derivatives with respect to loss###
  dLoss_dLogistic = (-y/y_pred) + ((1-y)/(1-y_pred))
  dLogistic_dLinear = y_pred*(1-y_pred)
  dLinear_dSlope = X
  ##computational graph
  dLoss_dSlope = np.sum(dLoss_dLogistic * dLogistic_dLinear * dLinear_dSlope) 
  dLoss_dBias = np.sum(dLoss_dLogistic * dLogistic_dLinear)
  
  slope = slope - learning_rate * dLoss_dSlope
  bias = bias - learning_rate * dLoss_dBias