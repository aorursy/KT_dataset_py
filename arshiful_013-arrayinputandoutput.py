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
arr=np.arange(5)
arr
np.save('myarray',arr)
arr=np.arange(10)
arr

np.load('myarray.npy')
#you can also save multiple arrays
arr1=np.load('myarray.npy')
arr1
arr2=arr
arr2
np.savez('ziparray.npz',x=arr1,y=arr2)
archive_array=np.load('ziparray.npz')
archive_array['x']
archive_array['y']
#you can also save an array in text files
arr=np.array([[1,2,3],[4,5,6]])
arr
np.savetxt('mytextarray.txt',arr,delimiter=',')
arr=np.loadtxt('mytextarray.txt',delimiter=',')
arr


