# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Training

df = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')
df = df.dropna()
data = df.values

X = data[:,0]
Y = data[:,1]

temp = np.dot(X,X)
B = np.dot(X,Y)
Beta = B/temp #Model

print(Beta)
#Testing

df_test=pd.read_csv('/kaggle/input/random-linear-regression/test.csv')
df_test = df_test.dropna()
data_test = df_test.values

X_test = data_test[:,0]
Y_test = data_test[:,1]

Y_cap = X_test * Beta #Predection
#Comparison

rmse = (((Y_cap - Y_test)**2).mean())**0.5
print(rmse)

plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
plt.plot(X_test, Y_test,'go',X_test, Y_cap,'r')
import random as rm
alpha = 0.0001
beta = rm.random()
m = len(Y_cap)
'''
for i in range(50):
    Y_cap = beta * X
    temp = np.dot((Y_cap - Y),X)
    beta = beta - alpha*temp/m
    print("Beta",beta)
    rmse = (((Y_cap - Y)**2).mean())**0.5
    print("RMSE =",rmse)
'''

rmse1 = 1000
rmse2 = 1000
rmse3 = 1000

while True:
    Y_cap = beta * X
    temp = np.dot((Y_cap - Y),X)
    beta = beta - alpha*temp/m
    #print("Beta",beta)
    rmse = (((Y_cap - Y)**2).mean())**0.5
    #print("RMSE =",rmse)
    rmse1 = rmse2
    rmse2 = rmse3
    rmse3 = rmse
    temp_mean = (rmse1 +rmse2 +rmse3 )/3
    if(abs(temp_mean - rmse) < 0.001):
        break
print("Beta",beta)
print("RMSE =",rmse)
temp_rmse = 0 
while True:
    Y_cap = beta * X
    temp = np.dot((Y_cap - Y),X)
    beta = beta - alpha*temp/m
    #print("Beta",beta)
    rmse = (((Y_cap - Y)**2).mean())**0.5
    #print("RMSE =",rmse)
    if(temp_mean > rmse):
        break
    temp_mean = rmse
print("Beta",beta)
print("RMSE =",rmse)
#Testing

Y_cap = X_test * beta #Predection
#Comparison

rmse = (((Y_cap - Y_test)**2).mean())**0.5
print(rmse)

plt.figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
plt.plot(X_test, Y_test,'go',X_test, Y_cap,'r')
