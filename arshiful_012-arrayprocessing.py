
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
import matplotlib.pyplot as plt
%matplotlib inline
points = np.arange(-5,5,0.1)
points
dx,dy=np.meshgrid(points, points)
dx
dy
z=(np.sin(dx) + np.sin(dy))
z
plt.imshow(z)
plt.imshow(z)
plt.colorbar()
plt.title('Plot for sin(x)+sin(y)')
A=np.array([1,2,3,4])
B=np.array([100,200,300,400])
condition=np.array([True,True,False,False])
answer=[(A_val if cond else B_val) for A_val,B_val,cond in zip(A,B,condition)]
answer
#shortcut using numpy
answer2=np.where(condition,A,B)
answer2
from numpy.random import randn
arr=randn(5,5)
arr
#useful tool to clean up your data
np.where(arr<0,0,arr)
arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
arr
arr.sum()
arr.sum(0)

arr.mean()
arr.std()
arr.var()
bool_arr=np.array([True,False,True])
bool_arr.any()
bool_arr.all()
#sort
arr=randn(5)
arr
arr.sort()
arr
countries=np.array(['France','Germany','USA', 'Russia','USA', 'Mexico', 'Germany'])
np.unique(countries)
np.in1d(['France','USA','Sweden'],countries)

