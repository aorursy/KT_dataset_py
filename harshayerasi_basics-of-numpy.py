# importing numpy
import numpy as np
np.int(2)
np.int(2.1)
np.int(2.8)
# Explore all the different datatypes


a=np.array([0,1,2,3,4,5])
a
print('The array is ',a,'and the type is',type(a),'and the elements datatype is',a.dtype)
a = np.array([0,1,2,3.0,4,5])
print('The array is ',a,'and the type is',type(a),'and the elements datatype is',a.dtype)
a = np.array([True,False])
print('The array is ',a,'and the type is',type(a),'and the elements datatype is',a.dtype)
# Setting datatype for the elements of an array.
np.array([0, 1, 2], dtype=np.uint8)
# Type casting to arrays
a=np.array([0,1,2,3,4,5])
a.astype(float) 
print('Demension of an array is',a.ndim)
b = np.array([[1,2,3],
              [4,5,6]])
print('The array is ',b,'and the type is',type(b),'and the elements datatype is',b.dtype,'with demensions',b.ndim)
c = np.array([[[1,2,3],[4,5,6]],
             [[7,8,9],[10,11,12]]])
print('The array is ',c,'and the type is',type(c),'and the elements datatype is',c.dtype,'with demensions',c.ndim)
print('Size of the array is',c.size)
d=np.array([[3,4,5,6],
           [7,8,9]])
# If the values are in consistent then it takes each block as one list of type object.
d.dtype
e=np.array(['a','b','c'])
e.dtype
# arange can only generate a single dimensional array.
f=np.arange(5)
print(f)
g=np.arange(5,55)
print(g)
h=np.arange(5,2)
print(h)
i=np.arange(5,105,5)
print(i)
j=np.arange(0,100,5).reshape(5,4)
print(j)
k=np.arange(0,100,5).reshape(2,2,5)
print(k)
# 2*2*5 = len(n)
k.ravel() # gives u the single dimension with all the elements
print(np.random.rand())
print(np.random.rand()*100)
print(np.random.randint(100))
print(np.random.randint(10,30))
np.random.randint(10,high=30,size=15).reshape(5,3)
print(np.floor(3.99))
print(np.ceil(3.09))
print(np.round(1.0))
print(np.round(1.1))
print(np.round(1.2))
print(np.round(1.3))
print(np.round(1.4))
print(np.round(1.5))
print(np.round(1.6))
print(np.round(1.7))
print(np.round(1.8))
print(np.round(1.9))
print(np.round(0.5))
print(np.round(1.5))
print(np.round(2.5))
print(np.round(3.5))
print(np.round(4.5))
print(np.round(5.5))
print(np.round(6.5))
print(np.round(7.5))
print(np.round(8.5))
print(np.round(9.5))
o=np.linspace(0,30,4) # Equally distributes
print(o)
p=np.linspace(0,20,4) # Equally distributes
print(p)
q=np.linspace(0,10,4) # Equally distributes
print(q)
r=np.array([2,3,4,5])
s=r # refering the same data with a different name
t= r.copy() # copying the data to a different variable.