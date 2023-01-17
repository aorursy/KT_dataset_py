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
import numpy as np 

a= np.array([1,2,3])

print(a)

print(type(a))
a = np.array([10,20,30,40,50])



print(a[1:4])
a[1]
import numpy as np 

x= np.array([    [1,2,3,4],  [5,6,7,8], [9,10,11,12]      ])

# print(x)

# print(x[1][2])

# print(x[:,1])

# print(x[2,:])

print(x[0:2, 1:3])
a= [1,2,3]

b=[[1,2,3] ,[4,5,6] ,[7,8,9]]
np.dot(a,b)
np.linalg.inv(b)
import numpy as np 

coeff = np.zeros((5, 5))

coeff
x= np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

x[1][2]

x[:,1]

x[2,:]
import numpy as np 

coeff = np.zeros((5, 5))

print(coeff)

import numpy as np 

coeff = np.zeros((5, 5))

print(coeff)

x=np.array([1,2,3,4,5])

a=[1,3]

for i in a:

        coeff[:, i] = x**3

print(coeff)
coeff[:,1]
coeff = np.zeros((3, 4))
coeff
x= np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
x
g=x+coeff
g=g+5
g
g=g-np.array([2,3,4,5])
g
g=g**2
g
5**3
f= np.array([[1,2],[3,4]])

print(f)

g=(f**3 -5)

print(g)
f
class Computer:

    def __init__(self):

        self.dog_name="happy"

        print("hello i am constructor")



    def __str__(self):

        self.dog_name="sad"

        return "hello"

    def customa(self):

        return "hello"+self.dog_name



  

    

a=Computer()

print(a)

print(a.customa())

# print(a)
