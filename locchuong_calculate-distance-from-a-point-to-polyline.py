# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""
Our input is theta as array and x0 as a number
The output will be a number, this is the result of f(x0)

step1: create a X array have a x element
step2: from zero to lenght of theta we append X, so we have enough x for theta array
step3: define a power function to calculate x^, with i as the formula above
step4: create a t_x array, t_x element is x^i
step5: multiply t_x and theta array the sum it and return the inner product

"""
def polynomial(theta,x):
    X = np.array([x])
    for i in range(len(theta)-1):
        X = np.append(X,x)
    
    power = lambda x: np.power(x,range(len(theta)))
    t_x  = power(X)
    return np.dot(theta.T, t_x)[0]
"""
with the polynomial you just write, let expand it, For a X array,
return a Y array as a result of every single element in X and Fx  
step1: create a empty array y
step2: for i in lenght of array X calculate f(X[i]) and append y with y
"""

def cal_poly(theta,x):
    y = np.array([])
    for i in range(len(x)):
        y = np.append(y,polynomial(theta,x[i]))
    return(y)
theta = np.flip(np.array([6,-9,1,-5,1,-3]), axis = 0).reshape((-1,1))
print(polynomial(theta,9))
"""
Now with theta
X is 100 points between -1 and 1 
Calculate Y
"""
X = np.linspace(-1,1,100)
Y = cal_poly(theta,X)

plt.scatter(X,Y)

"""
Create X array and Y, plot it  and you will see a smooth polyline
"""
theta = np.flip(np.array([6,-9,1,-5,1,-3]), axis = 0).reshape((-1,1))

X = np.linspace(-1,1,100)

Y = cal_poly(theta,X)

plt.plot(X,Y)
"""
NOW CREATE YOUR TAGETS, some point around the polyline, Just plus Y with a random point
"""
np.random.seed(0)
noise = np.random.normal(0, 2, len(Y))
Y_noise = Y+noise

plt.scatter(X,Y_noise,s = 10)

plt.plot(X,Y)

"""
I call point_in to descripe a point 's in polyline 
       point_out to descripe a point 's out polyline
So the point_in is the array of (X_step,Y_step) 
   the point_out is the array of (X,Y_noise)
Let's do it by 100 step in polyline and try again eith more step to check
"""
X_step = np.linspace(-1,1,100)
Y_step = cal_poly(theta,X_step)

point_out = np.stack((X,Y_noise),axis =1)
point_in = np.stack((X_step,Y_step),axis =1)
"""
Now let calculate distance between this lonely point and the evil polyline
The lonely point is the 20th point
"""
point_out[20]
"""
step1: create a array to store all result between the point in and the single point out
step2: calculate a as a distance between every point in and point out
step3: find the minimum in d_i, that will be what we looking for
"""

X_step = np.linspace(-1,1,100)
Y_step = cal_poly(theta,X_step)
d_i = np.array([])
for j in range(len(point_in)):
        a = np.sqrt(np.sum(np.power(point_out[20] - point_in[j],2)))
        d_i =  np.append(d_i,a)
minimum = d_i.min()
print(minimum)
"""
try it again
Now is 500 steps
"""

X_step = np.linspace(-1,1,500)
Y_step = cal_poly(theta,X_step)
d_i = np.array([])
for j in range(len(point_in)):
        a = np.sqrt(np.sum(np.power(point_out[20] - point_in[j],2)))
        d_i =  np.append(d_i,a)
minimum = d_i.min()
print(minimum)

"""
Okay let do it again
create a array of d to store distance array
for every point out find the minimum and store it
plot it and check
"""
X_step = np.linspace(-1,1,600)
Y_step = cal_poly(theta,X_step)
d =  np.array([])
d_i = np.array([])
for i in range(len(point_out)):
    for j in range(len(point_in)):
        a = np.sqrt(np.sum(np.power(point_out[i] - point_in[j],2)))
        d_i =  np.append(d_i,a)
    minimum = d_i.min()
    d = np.append(d,minimum)
    d_i = np.array([])

d
fig= plt.figure(figsize=(12,8))
axes= fig.add_axes([0.1,0.1,0.8,0.8])
step = np.arange(len(point_out))
axes.scatter(step,d)
axes.plot(step,d)
fig= plt.figure(figsize=(12,8))
axes= fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(X,Y_noise)
#axes.scatter(X,Y)
axes.plot(X,Y)
plt.show()
