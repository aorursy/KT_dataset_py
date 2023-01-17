#using numpy 

import numpy as np
#Creaing 1-d array



l=[10,20,30]

a=np.array(l)

a
#Creating 2-d array



l2=[[10,20,30],[40,50,60],[70,80,90]]



a2=np.array(l2)



a2
#Creating array using arange():



a1=np.arange(0,15,2)

a1
#Creating array using random



b=np.random.rand(5)   #rand-generates between 0 & 1(Exclusive)

b



#randn,randint also can be used
#Getting the dimensions of existing array 



a2.shape
#Changing the dimensions of existing array 

a1.reshape(2,4)



#Creating array of ones 

o =np.ones((5,3))

o
#Creating array of zeros

z=np.zeros(6)

z
#Creating identity matrix[Matrix which has equal number of rows and columns also all the non-diagonal elements are zero,also diagonal elments are 1]



i=np.eye(5)

i
#linspace-Returns equally spaced elements winthin a range

l=np.linspace(10,50,10)

l
#Inbulit arthimetic operations



print(a.min())

print(a.max())

print(a.sum())

print(a.argmax()) #index of maximum element

print(a.argmin()) #index of minimum elment
arr=np.random.randint(0,100,15)

arr=arr.reshape(5,3)

arr
arr[3][1]  #[][] row and column
arr[0,2]  #[row,column]
#slicing

arr[2:,0:2]
arr
#Broadcasting 



arr[:3,]=100

arr
[arr<50]
arr[arr<50]
a=np.random.randint(0,10,20)

a
b=np.random.randint(20,30,20)

b
a+b
a-b
a*b
a/b
np.sqrt(a)
np.dot(a,b) #dot product 
a
np.flip(a) #Reverses array