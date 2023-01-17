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
arr=np.arange(0,11)
arr
arr[8]

arr[1:5]
arr[0:5]
arr[0:5]=100
arr
arr=np.arange(0,11)
arr
slice_of_arr=arr[0:6]
slice_of_arr
slice_of_arr[:]=99
slice_of_arr
arr
#so now note that the changes have occured in the original array!! so changing data in the slice will change the data in the original array. avoids memory problem
#to get a copy you have to be explicit

arr_copy=arr.copy()
arr_copy
arr_2d=np.array(([5,10,15],[20,25,30],[35,40,45]))
arr_2d
arr_2d[1]
arr_2d[1,2]
arr_2d[1][2]
arr_2d
arr_2d[:2,1:]
arr_2d[1:,:2]
arr2d=np.zeros((10,10))
arr2d
arr_length=arr2d.shape[1]
arr_length
for i in range (arr_length):
    arr2d[i]=1
arr2d
for i in range (arr_length):
    arr2d[i]=i
arr2d
arr2d
arr2d[[2,4,7,9]]
