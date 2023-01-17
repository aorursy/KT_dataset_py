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
import numpy as np

mydata= np.array([[1,3,5,7],[2,4,6,8],[0,10,11,12]])

print (mydata.shape)

mydata2=mydata[:2,1:3]

print(mydata2)
print(mydata[0,1])
mydata2[0,0]=20

print(mydata[0,1])
row1=mydata[2,:]

row2=mydata[1:2,:]

print(row2,row1.shape)

print(row2,row2.shape)
mydata=np.array([0,2,4,6,8])

print(mydata[-1])

print(mydata[-4])
mydata=np.array([0,2,4,6,8])

print(mydata[-3:])
mydata=np.array([[0,2,4],[6,8,10],[12,14,16]])

x,y=mydata[:,:-1],mydata[:,-1]

print(x)

print(y)
data=np.arange(4)

print(data)
data=np.arange(4.0)

print(data)
data=np.arange(2,5)

print(data)
data=np.arange(3,9,2)

print(data)
mydata1=np.array([[3,4],[5,6]])

mydata2=np.array([[7,8],[0,1]])

print(mydata1+mydata2)

print(np.add(mydata1,mydata2))
print(np.sqrt(mydata1))
mydata1 = np.array([[2,4]])

mydata2 = np.array([[1],[3]])

mydata3 = np.array([[1,3], [2, 5]])

mydata4 = np.array([[0,1], [2, 3]])

print(np.dot(mydata1,mydata2))

print(np.dot(mydata3,mydata4))
print(np.dot(mydata1,mydata3))
mydata=np.array([1,3,5,7,0,2,4,6])

mydata=mydata.reshape((2,4))

print(mydata)
mydata=np.array([2,3,4,5,6])

print(mydata.shape)

print(mydata.shape[0])

mydata=mydata.reshape((mydata.shape[0],1))

print(mydata.shape)

print(mydata)
import matplotlib.pyplot as plt

mydata1=np.array([0,1,2])

mydata2=np.array([3,5,7])

plt.plot(mydata)

plt.xlabel('data specification for x axis')

plt.ylabel('data specification for y axis')

plt.show()
myData1 = np.array([2, 4, 6])

myData2 = np.array([1, 3, 5])

plt.scatter(myData1, myData2)

plt.xlabel('data specification for x axis')

plt.ylabel('data specification for y axis')

plt.show()

import pandas as pd

myData = np.array([[0, 2, 4], [1, 3, 5]])

row_names = ['row 1', 'row 2']

col_names = ['first', 'second', 'third']

dataframe = pd.DataFrame(myData, index=row_names, columns=col_names)

print(dataframe)
