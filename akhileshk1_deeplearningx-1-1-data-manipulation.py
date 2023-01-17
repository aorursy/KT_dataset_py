# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np
x=np.arange(10)

x
print(x.size)

x.shape
x = x.reshape(5, 2)

x
print(x.reshape(-1,2))

x.reshape(5,-1)
np.empty((5,2))
print(np.zeros((5,2)))

np.ones((5,2))
np.random.normal(0,1,size=(5,2))
np.array([[2, 1],[1, 1],[3, 1], [2, 5],[1, 1]])
x = np.array([1, 2, 4, 8])

y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
np.exp(x)
x = np.arange(10).reshape(5,2)

y = np.array([[2, 1],[1, 1],[3, 1], [2, 5],[1, 1]])

x,y,np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
x==y
x.sum()



np.sum(x)
a = np.arange(6).reshape(3, 2)

b = np.arange(2).reshape(1, 2)

a, b
a+b
x,x[-1], x[1:4]
x[1,1]=5

x



# changes the 1,1 element to 5