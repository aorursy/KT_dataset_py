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
arr=np.arange(50).reshape((10,5))
arr
arr.T
arr
np.dot(arr.T,arr)
arr3d=np.arange(50).reshape(5,5,2)
arr3d
arr3d.transpose((1,0,2))
arr=np.array([[1,2,3]])
arr
arr.swapaxes(0,1)
