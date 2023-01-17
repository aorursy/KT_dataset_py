# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np
np.arange(10) # creates an array with values from 0 to 9.
np.arange(1,10) #creates an array that starts at 1
np.arange(0, 10, 2) #the third argument is a stepping argument, i.e., how big the increments are
np.arange(1,20,3,dtype=np.float64)
x = np.array([1,3,5,7])

x
type(x)
np.array({(1,3,5),(7,9,11),(13,15,17)}) #(a,b,c) is a sequence, now this 

#is a 2-D array with columns and rows
np.array([(1,3,5),(7,9,11),(13,15,17)]).ndim
np.zeros((2,3), dtype=np.int16) #zeros specifies the size of desired matrix, in this case

#it is a 2x3 matrix.
np.ones((3,3), dtype=np.int16)
np.empty((2,2,4)) #keyword empty is to create random values (this is a multidimensional array)
np.linspace(2,4,10) #better for floating point arguments because of precision

#as opposed to arange, it ends in 4. 
np.random.random((3,3))
ds=np.arange(1,10,3)

print(ds)

ds.ndim
threeD=np.arange(1,30,2).reshape(3,5) #converts a one-dimensional array 

#to a higher dimensional array

print(threeD)
threeD.shape
ds.size
ds.dtype
ds.itemsize
a=np.arange(5)

b=np.array([2,4,0,1,2])

a
diff=a-b

diff
b**2
2*b
np.sin(a)
np.sum(a)
np.max(a)
np.min(a)
b>2 #compares ALL values in b to 2
a*b #matrix multiplication
x=np.array([[1,1],[0,1]])

y=np.array([[2,0],[3,4]])

x*y
x.dot(y) #dot product. same as np.dot(x,y)
x.sum() #argument is to define the axis we want to sum up

#axis=0 is getting the sum of colum, axis=1 is summing the row values
x.sum(axis=1)

x.sum(axis=0)
z = np.random.random((3,4))

z
print(np.mean(z), np.median(z), np.std(z))
data_set=np.random.random((2,3))

data_set
np.reshape(data_set,(3,2))
np.reshape(data_set,(6,1))
np.reshape(data_set,(6))
np.ravel(data_set)
data_set=np.random.random((5,10))

data_set
data_set[1]
data_set[1][0]
data_set[1,0]
data_set[2:4]
data_set[2:4,0]
data_set[2:4,0:2]
data_set[:,0]
data_set[2:5:2] #get values from row 2 to 3, but in increments of 2
data_set[:2] #just show values at every 2 idices
data_set[2:4,::2]