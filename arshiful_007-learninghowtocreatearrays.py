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
my_list1=[1,2,3,4]
my_array1=np.array(my_list1)
my_array1
my_list2=[11,22,33,44]
my_lists=[my_list1,my_list2]
my_array=np.array(my_lists)
my_array

my_array.shape
my_array.dtype
np.zeros(5)
my_zeros_array=np.zeros(5)
my_zeros_array.dtype
np.ones([5,5])
np.empty(5)
np.eye(5)
np.arange(5)
np.arange(5,50,2)