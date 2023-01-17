# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import fmin_bfgs
import matplotlib
import matplotlib.pyplot as plot
#from sklearn import preprocessing
#from matplotlib import style
import pylab
import datetime
import re

# Input data files are available in the "../input/" directory.
df = pd.read_csv('../input/train.csv',usecols = ['is_booking','srch_adults_cnt','srch_destination_id',\
'srch_ci','srch_co','hotel_cluster'],chunksize=1000)
df = pd.concat(df, ignore_index=True)

df = df.groupby(['is_booking']).get_group(1)

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dfx = df.ix[:,'hotel_cluster']
ylabel = dfx.value_counts()
ylabel = ylabel.index
y = dfx.as_matrix()

dfx = df.ix[:,'srch_adults_cnt']
x1 = dfx.as_matrix()
mu = np.mean(x1)
s = np.amax(x1)-np.amin(x1)
x1 = (x1 - mu)/s

dfx = df.ix[:,'srch_destination_id']
x2 = dfx.as_matrix()
mu = np.mean(x2)
s = np.amax(x2)-np.amin(x2)
x2 = (x2 - mu)/s

dfx = df.ix[:,'srch_ci']
dfx = pd.to_datetime(dfx)
ci = dfx.dt.year*365 + dfx.dt.month*30 + dfx.dt.day

dfx = df.ix[:,'srch_co']
dfx = pd.to_datetime(dfx)
co = (dfx.dt.year)*365 + (dfx.dt.month)*30 + dfx.dt.day

x3 = co - ci
x3 = x3.as_matrix()
mu = np.mean(x3)
s = np.amax(x3)-np.amin(x3)
x3 = (x3 - mu)/s

x0 = np.ones(len(y))

X = np.vstack((x0,x1,x2,x3))

print(X.shape)
X = np.transpose(X)

m, num_features = X.shape
print(X.shape, m, num_features)
lmd = 1.0
all_theta = []

initial_theta = np.random.rand(num_features,1)
theta = fmin_bfgs(costfunction, initial_theta , fprime = gradient, args = (X,y==0,lmd))
print(theta)
def sigmoid(z):

  h = np.reciprocal(1.0 + np.exp(-z))
  return h


def costfunction(theta,X,y,lmd):

  #z = np.dot((np.transpose(X)),theta)
  #print(X.shape, theta.shape)
  z = np.dot( X , theta )
  h = sigmoid(z)
  m = len(y)
  y = np.reshape( y , (-1,1))
  n = len(theta)
  J = (1.0/m)*( - np.dot(np.transpose(y),np.log(h)) \
                - np.dot(np.transpose(1 - y),np.log(1 - h)) ) \
      + (0.5*lmd/m)*( np.dot( np.transpose( theta[1:m] ) , theta[1:m]  )  )
  return J

def gradient(theta,X,y,lmd):

  m = len(y)
  n = len(theta)
  #print(X.shape,theta.shape)
  #theta = np.reshape( theta, (-1,1) )
  #print(X.shape,theta.shape)
  z = np.dot(X , theta)
  #z = np.dot( X , theta)
  h = sigmoid(z)
  #print(h.shape, y.shape, X.shape)
  #rint(h.shape,y.shape)
  grad = np.dot( np.transpose(X) , (h-y) )/m
  temp = theta
  temp[0] = 0
  grad = grad + (lmd/m)*temp
  return  grad

theta = fmin_bfgs(costfunction, initial_theta , fprime = gradient, args = (X,y==0,lmd))
print(theta)
theta = fmin_bfgs(costfunction, initial_theta , fprime = gradient, args = (X,y==0,lmd))
all_theta.append(theta)
theta = fmin_bfgs(costfunction, initial_theta , fprime = gradient, args = (X,y==1,lmd))
all_theta.append(theta)
theta = fmin_bfgs(costfunction, initial_theta , fprime = gradient, args = (X,y==2,lmd))
all_theta.append(theta)
theta = fmin_bfgs(costfunction, initial_theta , fprime = gradient, args = (X,y==3,lmd))
all_theta.append(theta)
theta = fmin_bfgs(costfunction, initial_theta , fprime = gradient, args = (X,y==4,lmd))
all_theta.append(theta)
theta = fmin_bfgs(costfunction, initial_theta , fprime = gradient, args = (X,y==5,lmd))
all_theta.append(theta)
theta = fmin_bfgs(costfunction, initial_theta , fprime = gradient, args = (X,y==5,lmd))
all_theta.append(theta)
