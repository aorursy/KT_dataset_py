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
np.random.seed(99)

arr = np.random.randn(1000)

arr[arr<0.9] = np.nan
arr
sparr = pd.SparseArray(arr)

sparr
ser = pd.Series(sparr, name='name')
ser.head()
df = pd.DataFrame([ser])

df.shape
df.info()
df.to_dense().info()