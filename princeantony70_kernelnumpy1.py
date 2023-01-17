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
print("hello")
import numpy as np 



mylist =[1,2,3,]



myarray = np.array(mylist)

print (myarray)

print(type(myarray))
import numpy as np 

my_matrix =[[1,2,3],[3,4,5],[5,6,7]]



mymatc = np.array(my_matrix)

mymatc
import numpy as np 

np.arange(0,10)
import numpy as np 



arr = np.arange(0,10,2)
import numpy as np 

np.zeros(5)
import numpy as np 

np.zeros((5,5))
import numpy as np 

np.ones((3,2))
import numpy as np 

print(np.linspace(0,9,10))

print(np.linspace(0,5,10))

#it share the range into equal gap as per last parameter
import numpy as np 

np.eye(4)

np.eye(5)
import numpy as np

print(np.random.rand(5))

print(np.random.rand(5,5))
print(np.random.randn(2))

print(np.random.randn(4,4))
print(np.random.randint(1,100))

print(np.random.randint(1,100,10))

import numpy as np 

np.random.randint(1,100,5)

arr = np.arange(25)

print(arr)

rarr = np.random.randint(0,50,10)

print(rarr)

print(arr.reshape(5,5))

print(rarr.max())

print(rarr.min())

print(rarr.argmax())

print(arr.shape)
arr.dtype