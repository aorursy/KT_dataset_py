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
import numpy as np

my_list=[1,2,3]
arr = np.array(my_list)  #cast my_list into an array using numpy(np) 
print(arr)

import numpy as np

my_matrix=[[1,2,3,4,5,6],[4,6,7,4,5,6],[56,56,5,3,2,1],[4,6,7,6,7,8]]
print(my_matrix)

matrix=np.array(my_matrix)
print(matrix)

a=np.arange(0,100,5)
print(a)

import numpy as np
np.random.rand(4,4)

np.random.randn(4,4)

hahaha = np.random.randint(0,50,10)
print('My random array is {num}'.format(num=hahaha))
