import numpy as np
#one dimensional array

data1=[5,6,7,8,9]

arr1=np.array(data1)

print(arr1)

#multi dimansional array

data2=[[1,2,3],[4,5,6]]

arr2=np.array(data2)

print(arr2)
arr3=np.arange(10)

print(arr3)



arr4=np.arange(2,20,2)

print(arr4)



arr5=np.arange(20,2,-3)

print(arr5)
arr6=np.zeros(10)

print(arr6)



arr7=np.zeros(5,dtype='int')

print(arr7)
arr8=np.ones(15)

print(arr8)

arr8=np.ones(10,dtype='int')

print(arr8)

arr8=np.ones((4,3),dtype='int')

print(arr8)
arr9=np.full((3,5),1.5)

print(arr9)
arre=np.empty(5)

print(arre)

arre=np.empty((2,3),dtype='int')

print(arre)
arr10=np.eye(5)

print(arr10)
arr11=np.linspace(1,5,10)

print(arr11)
arr=np.eye(5)

print(arr)

print("dimensions ",arr.ndim)

print("shape", arr.shape)

print("size",arr.size)

print("data type",arr.dtype)

print("size of each element",arr.itemsize)

print("total size",arr.nbytes)
arr=np.random.randint(100)

print(type(arr))

print(arr)
arr=np.random.randint(1,100,10)

print(arr)



arr=np.random.randint(1,100,(10,2))

print(arr)

print(type(arr))
number=np.random.rand()

print(number)

print(type(number))
arr=np.random.rand(5)

print(arr)

print(type(arr))



arr=np.random.rand(5,4)

print(arr)





arr=np.random.rand(5,4,2)

print(arr)
