import numpy as np

import time

import sys

S=range(1000)

print(sys.getsizeof(S)*len(S))

N=np.arange(1000)

print(N.size*N.itemsize)

size=1000000

L1=range(size)

L2=range(size)

A1=np.arange(size)

A2=np.arange(size)

start=time.time()

result=[(x,y) for x,y in zip(L1,L2)]

print((time.time()-start)*1000)

start=time.time()

result=A1+A2

print((time.time()-start)*1000)

result
print(S)
import numpy as np

a=np.array([5,6,7])

a
a.dtype
b=np.array(["smiuta","poo","foo"])

b.dtype
a.itemsize
a.dtype.name
b.dtype.name
a.size
x=np.array([[5,6],[7,8]],dtype=complex)

x
np.zeros((2,3))
np.ones((2,3))
np.empty((3,3))
np.arange(10,21,4)
np.arange(1,2,0.25)
np.linspace(0,2,9)
#1D array

np.arange(5)
#2D array

np.arange(12).reshape(3,4)
#3D array

np.arange(24).reshape(2,3,4)
A=np.arange(12).reshape(2,6)

B=np.arange(18).reshape(2,9)
A
B
A=np.array([[4,5],[6,7]])

B=np.array([[7,8],[9,10]])
#Element wise product

A*B
#Matrix product



A@B
a=np.arange(24).reshape(6,4)

a
a.sum()
a.min()
a.max()
#sum 0f each column

a.sum(axis=0)
#Sum of each row

a.sum(axis=1)
#cumulative sum along each row

a.cumsum(axis=1)
B=np.arange(6)

np.exp(B)
np.sqrt(B)
C=np.linspace(2,16,6)

np.add(B,C)
T=np.arange(12).reshape(3,4)

T
T.size
#Each row in the second column of T

T[0:3,1]
#Equivalent to previous

T[:,1]
#Each column with 2nd and 3rd row



T[1:3,:]
#The last row

T[-1]
#Returns the array flattened

T.ravel()
#Transpose of array

T.T
T.T.shape
T.shape
a=np.array([[1,2],[3,4]])

a
b=np.array([[5,6],[7,8]])

b
np.vstack((a,b))
np.hstack((a,b))
# split a into 3

x=np.arange(12).reshape(4,3)

np.hsplit(x,3)
x
np.vsplit(x,2)
np.hsplit(x,5)
np.array_split(x,5)
# No copy at all

#a=np.array([[1,2],[3,4]])

a=np.arange(24).reshape(6,4)

a

b=a #no new object is created at all

b is a # b and  a are two names for the same object
#View or shallow copy

c=a.view()

c is a
c.flags.owndata
c
a=a.reshape(2,12) #C is not changes by reshape()

c
c=c.reshape(8,3) # a  is not chnage

a
c
#

c[1,2]=67

c
a #Value in a also get changes as we change in c
d=a.copy() #complete copy

d
d.base is a #d soes not share anything with a
d[0,0]=1000 #no change in a
d
a
from matplotlib import pyplot as plt

x=np.arange(1,50,2)

y=3*x+10

plt.plot(x,y)
#Sine plot

y=np.sin(x)

plt.plot(x,y)
#Bar plot



x=[1,2,3]

y=[10,20,30]

x1=[4,5,6]

y1=[7,8,9]

plt.bar(x,y,color='g')

plt.bar(x1,y1)
#Histogram



x=np.array([1,2,4,56,78,89,34,66.89,76,54])

y=[20,40,60,80,100]

plt.hist(x,bins=y)
x=np.array([1,2,3,4,5])

np.save('out',x)

b=np.load('out.npy')

b