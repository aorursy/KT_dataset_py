# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
a=2+3
print(a)

a=np.array([1,2,3,4,5])
print(a)

a=np.array([1,2,'helo',3.567])
print(a)
print(a[0:3])#-------------------------------slicing


#using the inbuilt syntax and creation of the arrays using numpy

#using the linspace
'''start= start value of the array 
   stop=end value of the array
   num=number of samples to be present in the array
   endpoint=ture:to iclude the stop value
           =false:to exclude the stop vlaue
   retstep=true:show the step value in output
           =False:not to show step value in output'''

a=np.linspace(start=1,stop=10,num=5,endpoint='true',retstep='true')
print(a)


#numpy.arange
a=np.arange(start=1,stop=10,step=2)
print(a)

#numpy.ones((shape),datatype=float)

a=np.ones((3,3),)
print(a)

a=np.ones((3,3),int)
print(a)

#numpy.zeros((shape),datatype=flaot)
a=np.zeros((3,3),)
print(a)


#numpy.random.rand((shape))
a=np.random.rand(2,3)
print(a)

#numpy.logsapce(start,stop,num=50(default value),endpoint,base=10(default value))
a=np.logspace(2,30,10,'true',10)
print(a)




a=np.arange(1,19).reshape(3,6)
print(a)
a.shape #----------------------gives the dimension
print('dimension=',a.shape)


b=np.arange(1,20,2).reshape(2,5)
print(b)


"""
a=np.arange(1,11).reshape(3,6)
print(a)
shows the error that it cant reshape a size of 10 into 3,6 array
"""
