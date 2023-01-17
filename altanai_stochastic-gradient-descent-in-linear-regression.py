# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%matplotlib inline

import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

import random

from scipy import stats

from scipy.optimize import fmin
f = lambda x: x*2+17+np.random.randn(len(x))*10
x = np.linspace(-1,20,1000)

plt.plot(x,f(x))

plt.xlim([-1,20])

plt.ylim([0,100])

plt.show()
x = np.random.random(500000)*100

y = f(x) 

m = len(y)
from random import shuffle



x_shuf = []

y_shuf = []

index_shuf = list(range(len(x)))

shuffle(index_shuf)

for i in index_shuf:

    x_shuf.append(x[i])

    y_shuf.append(y[i])
h = lambda theta_0,theta_1,x: theta_0 + theta_1*x
cost = lambda theta_0,theta_1, x_i, y_i: 0.5*(h(theta_0,theta_1,x_i)-y_i)**2
theta_old = np.array([0.,0.])

theta_new = np.array([1.,1.]) # The algorithm starts at [1,1]

n_k = 0.000005 # step size
iter_num = 0

s_k = np.array([float("inf"),float("inf")])

sum_cost = 0

cost_list = []
for j in range(10):

    for i in range(m):

        iter_num += 1

        theta_old = theta_new

        s_k[0] = (h(theta_old[0],theta_old[1],x[i])-y[i])

        s_k[1] = (h(theta_old[0],theta_old[1],x[i])-y[i])*x[i]

        s_k = (-1)*s_k

        theta_new = theta_old + n_k * s_k

        sum_cost += cost(theta_old[0],theta_old[1],x[i],y[i])

        if (i+1) % 10000 == 0:

            cost_list.append(sum_cost/10000.0)

            sum_cost = 0   

            

print("Local minimum occurs where:")

print("theta_0 =", theta_new[0])

print("theta_1 =", theta_new[1])
iterations = np.arange(len(cost_list))*10000

plt.plot(iterations,cost_list)

plt.xlabel("iterations")

plt.ylabel("avg cost")

plt.show()