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
thisdict={'x':10,'y':15,'z':20}

print('An item: %d' % thisdict['x'])
mydict={'x':4}

print ('An item:%d' %mydict['x'])
print('keys are :%s' %thisdict.keys())
print('values are : %s' % thisdict.values())
for i in thisdict.keys():

    print (thisdict[i])
import numpy as np

myarray= np.zeros(10,int)

print(myarray)
myarray=np.zeros(5,str)

print (myarray)
myarray=np.zeros(5)

print(myarray)
a=np.zeros((2,2))

print(a)
a=np.random.random(10)

print(a)
e=np.random.random((2,2))

print(e)
mydata=np.array([5,10,15])

print(mydata)

print(mydata.shape)

print(mydata[0],mydata[1],mydata[2])
mydata[0]=5

print(mydata)
mydata=np.array([[10,20,30],[40,50,60]])

print(mydata.shape)

print(mydata.shape[0])

print(mydata.shape[1])

print(mydata[0,0],mydata[0,1],mydata[1,0])
def Multiply(a,b):

    return a*b

x=Multiply(2,4)

print(x)