# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pylab as plt
from platform import python_version
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# for course 
# https://github.com/JunChiehWang/Statistical_Learning_Stanford
# 
print("python version: ",python_version())
print("numpy version: ",np.__version__)
print("pandas version: ",pd.__version__)
print("matplotlib version: ",matplotlib.__version__)

version='20190127'
print('sigmoid_function version:',version)
print('date and time: ',datetime.datetime.now())
# plot 3 figures with different bo range

fig, ax = plt.subplots(nrows=3, ncols=1,figsize=(10,6))
b1=1.e-2
x = np.linspace(-5000,5000,61)

# fig[0]
bo_array = np.linspace(-40,40,3)
for bo in bo_array:
    #print('bo=',bo)
    z = bo + b1*x
    p=1/(1+np.exp(-z))
    ax[0].plot(x,p,'--o',label='bo=%f' %bo)
    ax[0].legend(loc=[1,0])
    ax[0].set_ylabel('p')
    ax[0].set_xlabel('x')
    ax[0].set_title('b1=%f, change bo' %b1)
    
# fig[1]
bo_array = np.linspace(-20,20,3)
for bo in bo_array:
    #print('bo=',bo)
    z = bo + b1*x
    p=1/(1+np.exp(-z))
    ax[1].plot(x,p,'--o',label='bo=%f' %bo)
    ax[1].legend(loc=[1,0])
    ax[1].set_ylabel('p')
    ax[1].set_xlabel('x')
    ax[1].set_title('b1=%f, change bo' %b1)
    
# fig[2]
bo_array = np.linspace(-10,10,3)
for bo in bo_array:
    #print('bo=',bo)
    z = bo + b1*x
    p=1/(1+np.exp(-z))
    ax[2].plot(x,p,'--o',label='bo=%f' %bo)
    ax[2].legend(loc=[1,0])
    ax[2].set_ylabel('p')
    ax[2].set_xlabel('x')
    ax[2].set_title('b1=%f, change bo' %b1)
    
plt.tight_layout(pad=2)    
# plot 3 figures with different bo range

fig, ax = plt.subplots(nrows=3, ncols=1,figsize=(10,6))
bo=-40
x = np.linspace(-20,20,101)

# fig[0]
b1_array = np.linspace(-400,400,3)
for b1 in b1_array:
    #print('b1=',b1)
    z = bo + b1*x
    p=1/(1+np.exp(-z))
    ax[0].plot(x,p,'--o',label='b1=%f' %b1)
    ax[0].legend(loc=[1,0])
    ax[0].set_ylabel('p')
    ax[0].set_xlabel('x')
    ax[0].set_title('bo=%f, change b1' %bo)
    
# fig[1]
b1_array = np.linspace(-5,5,3)
for b1 in b1_array:
    #print('b1=',b1)
    z = bo + b1*x
    p=1/(1+np.exp(-z))
    ax[1].plot(x,p,'--o',label='b1=%f' %b1)
    ax[1].legend(loc=[1,0])
    ax[1].set_ylabel('p')
    ax[1].set_xlabel('x')
    ax[1].set_title('bo=%f, change b1' %bo)
    
# fig[2]
b1_array = np.linspace(-3,3,3)
for b1 in b1_array:
    #print('b1=',b1)
    z = bo + b1*x
    p=1/(1+np.exp(-z))
    ax[2].plot(x,p,'--o',label='b1=%f' %b1)
    ax[2].legend(loc=[1,0])
    ax[2].set_ylabel('p')
    ax[2].set_xlabel('x')
    ax[2].set_title('bo=%f, change b1' %bo)
    
plt.tight_layout(pad=2)    
# plot with different bo and b1
fig, ax = plt.subplots(figsize=(10,6))
x = np.linspace(-20,20,6401)

#
bo=-0.2
b1=0.2
z = bo + b1*x
p=1/(1+np.exp(-z))
ax.plot(x,p,'--o',label='bo={first}, b1={last}'.format(first=bo, last=b1))

#
bo=-2
b1=2
z = bo + b1*x
p=1/(1+np.exp(-z))
ax.plot(x,p,'--o',label='bo={first}, b1={last}'.format(first=bo, last=b1))

#
bo=-50
b1=50
z = bo + b1*x
p=1/(1+np.exp(-z))
ax.plot(x,p,'--o',label='bo={first}, b1={last}'.format(first=bo, last=b1))

#
bo=-100
b1=100
z = bo + b1*x
p=1/(1+np.exp(-z))
ax.plot(x,p,'--o',label='bo={first}, b1={last}'.format(first=bo, last=b1))

ax.legend()

plt.tight_layout(pad=2)    
