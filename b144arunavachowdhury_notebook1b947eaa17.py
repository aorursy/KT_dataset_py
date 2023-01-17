import numpy as np

#creating 1d array

oned_array=np.array([2,3,5,8])

print(oned_array)
#creating 2d array

twod_array=np.array([[12,34,56,22],[43,56,11,77]])

print(twod_array)
#creating a boolean array

bool_array=np.array([1,2,33,0,"Arunava",False,True,"",33],dtype=bool)

print(bool_array)
#Extracting odd numbers from a 2D numpy array

a=np.array([[1,32,13,45,11,55],[2,33,22,45,12,67]])

print(a[a%2==1])
#Replace numpy array items satisfying a condition with a default value

a=np.array([[1,32,13,45,11,55],[2,33,22,45,12,67],[33,55,77,99,123,1]])

a[a%2==1]=-1 # <array name>(<condition>)=<value to be put instead>

print(a) #here e replace all the odd numbers by -1
#Get the common items between 2 numpy arrays

array1=np.array([0,34,56,7,23,67,22,90])

print("Array1: ",array1)

array2 =[22,45,67,7]

print("Array2: ",array2)

print("Common values between two arrays:",np.intersect1d(array1,array2))

array1=np.array([[2,3,4,5],[33,45,12,44]])

print("Array1: ",array1)

array2=np.array([[1,2,33],[45,12,5]])

print("Array2: ",array2)

print("Common values between two arrays:",np.intersect1d(array1,array2))
#Remove from an array all its common items with another array

a=np.array([33,33,13,44,55,66,77,55,12,23,21,34,59])

b=np.array([21,34,55,77])

sidx=b.argsort()

out=a[b[sidx[np.searchsorted(b,a,sorter=sidx)]]!=a]

print(out)
#Get the indexes where the elements match in the 2 numpy arrays

a=np.array([33,44,55,66,77,88,99,111,222])

b=np.array([12,44,56,67,77,87,99,111,221])

print("The Indexes where the elements match in the 2 arrays:")

np.where(a==b)