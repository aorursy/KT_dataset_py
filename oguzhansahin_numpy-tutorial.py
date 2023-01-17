#The first thing that we have to do is importing library.



import numpy as np
data_list = [1,2,3,4] 

data_list
#Creating NumPy array



array = np.array(data_list)

array
data_list2 = [[1,2,3],[4,5,6],[7,8,9]]



array2 = np.array(data_list2)

array2
array3 = np.array([1,2,3,4])

array3
#Indexing



array3[0]
array2[0,2]
np.arange(0,20) #0 included,20 not.
np.arange(0,100,5) 
np.zeros(10)
np.zeros((4,4))
np.ones(20)
np.ones((4,4,3))
np.linspace(0,100,5) #Divides values from 0 to 100 into 5 parts.
np.linspace(0,1,5)
np.eye(6) #unit matrix
np.random.randint(10) #0 included ,10 not.
np.random.randint(0,10)
np.random.randint(0,10,3) #creating 3 random numbers between 0 and 10.
np.random.rand(5)
arr = np.arange(25)

arr
arr.reshape(5,5)
newArray = np.random.randint(1,100,10)

newArray

newArray.max()
newArray.min()
newArray.mean()
newArray.argmin() #Index of minimum number
newArray.argmax()
array = np.arange(0,50,5)

array
array[1]
array[1:7] #1 to 7(7 not included)
array[:8] #0 to 8(8 not included)
array[1:]
array[::]  #all elements
array [::9] #array[0] and array[9]
random = np.arange(0,10)

random
random > 3
boolean = random > random.mean()

boolean

random[boolean]
array1=np.random.randint(0,10,5)

array1
array2 = np.random.randint(0,10,5)

array2
array1+array2
array1*array2
array1+10
array2**2