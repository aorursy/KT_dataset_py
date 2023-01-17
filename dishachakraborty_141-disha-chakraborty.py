#1. Create 1D,2D and boolean array using numpy

# 1D array

import numpy as np

a=np.array([1,2,3,4])

print(a)

print("\n")

# 2D array

b=np.array([[1,2,3],[4,5,6]])

print(b)

print("\n")

#boolean array

boolean=np.array([1,0,1,0,0,1],dtype=bool)

print(boolean)

print("\n")

#2. Extract odd numbers from 2D array using numpy

arr=np.array([[5,6,7],[8,9,10]])

for i in arr:

    for j in i:

        if(j%2!=0):

            print(j,end=" ")

print("\n")

#3. How to replace items that satisfy a condition with any other value

arr=np.array([3,4,9,2,15,23,21,80,99])

arr[arr%3==0]=0

print(arr)

print("\n")

#4. Common values between two numpy arrays

a=np.array([2,4,5,7,3])

b=np.array([5,1,7,9,2])

c=np.intersect1d(a,b)

print(c)

print("\n")

#5. We will remove common items from one any array 

a=np.array([1,3,7,6,8,9])

b=np.array([1,4,5,7,9,3])

c=np.setdiff1d(a,b)

print("The first array after removing common items is ",c)

d=np.setdiff1d(b,a)

print("The second array after removing common items is ",d)

print("\n")

#6. Index of common element

arr1=np.array([0,3,4,5,6,7])

arr2=np.array([1,2,8,9,6,11])

index=np.where(arr1==arr2)

print("The index of the common element is ",index[0][0])



            

        


