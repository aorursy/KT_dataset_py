# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
list1 = [1,2,3,4]
array1 = np.array(list1)
array1
list2 = [5,6,7,8]
array2 = np.array(list2)
array2
big_list = [list1,list2]
array = np.array(big_list)
array.shape
zero = np.zeros([3,2])
one = np.ones(6)
zero
ide = np.eye(3)
ide
arange = np.arange(5,50,4)
arange
mat = np.array([[1,2,3,4],[1,2,3,4]])
mat1 = mat*mat
power = mat**3
power
#11 numbers equidistant between 0 and 11
arr12 = np.linspace(0,10,11)
arr12[0:7]
arr12[0:3] = 100
arr12
slice_of_arr12 = arr12[:3]
slice_of_arr12
slice_of_arr12[:] = 99
slice_of_arr12
#If u do change in slice, it affects the parent array
arr12
#copy() returns a separate copy
arr_copy = arr12.copy()
arr_copy
arr_new = np.array(([0,1,2],[3,4,5]))
arr_new[0]
arr_new[:3,1:3]
next_arr = np.arange(50).reshape(5,10)
next_arr.T
np.dot(next_arr,next_arr.T)
np.sqrt(next_arr)
np.exp(next_arr)
A = np.random.randn(10)
B = np.random.randn(10)
A
np.add(A,B)
A = np.array([0,1,2,3])
B = np.array([100,200,300,400])
cond = np.array([True ,False ,True ,False])

answer = [(A_val if condition else B_val) for A_val,B_val,condition in zip(A,B,cond)]
answer
#same can be done as below
answer2 = np.where(cond,A,B)
answer2
from numpy.random import randn
randoms = randn(2,3)
randoms
test_arr = np.array([.01,-8,-5,4,0,3])
relu = np.where(test_arr > 0,test_arr,0)
relu
tarray = np.array([[1,4,2],[5,8,3],[0,4,1]])
tarray.sum()
#0 stands for axis x/y
tarray.sum(0)
tarray.std()
#0 stands for index
tarray.sort(0)
tarray
country = np.array(['US','CN','IN','CN','SZ','US'])
np.unique(country)
