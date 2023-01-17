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
np.arange(2,11).reshape(3,3)
a = np.empty(10)

a[5] = 11

a
np.arange(2,38)
a = np.arange(12,38)

b = a[::-1]

b
a = np.arange(1,5)



np.asfarray(a)
a = np.ones((5,5))

a[1:-1, 1:-1] = 0

a
a = np.zeros((3,3))

a[1:-1,1:-1] = 1

a
a = np.zeros((8,8))

a[arange(0,64,2)] = 1

a
a = np.zeros((8,8))

a[1::2, ::2] = 1

a

tup = (1,2,3,4,5,6)

a = np.array(tup)

a.resize(2,3)

a

a = np.arange(10,40,10)

x = np.append(a,[20,40])

x
a = np.empty((3,4), dtype = "int")

a[:] = 6 # a = np.full((3,4) , 6)

a
a = np.array([1,3,4])

b = (2*a / 3)

a = np.append(a,a)

a


