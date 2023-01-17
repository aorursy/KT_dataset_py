import numpy as np

firstArray = np.array([1,2,3,4,5,6,7,8])

print(type(firstArray))
a1 = np.array([1,2,3])

# multiple each member of a1 by 2
print(a1*2)

#get sin of each member of a1
print(np.sin(a1))

#for each member of a1, see if they are greater than 2
print(a1<2)
#initialize a second array
b1 = np.array([4,5,6])

# add each member of a1 to b1's member in relative index
print(a1+b1)

# substract each member of a1 from b1's member in relative index
print(a1-b1)

#just opposite of above
print(b1-a1)

# power calculation where a1 is base and b1 is power
print(a1**b1)

# get square roots of each member of an array
arr5 = np.array([9,16,25,36,49,64])
print(np.sqrt(arr5))
# get an array of numbers that starts with 10, goes until 50 by 5
print(np.arange(10,50,5))

# get an array that divides interval of 1 and 2 to 5
print(np.linspace(1, 2, 5))
myArray = np.array([1,2,3,4,5,6,7,8])
myMatrix = myArray.reshape(2,4)
print(myMatrix)

print("shape of our matrix : " + str(myMatrix.shape))
print("dimensions of our matrix: " + str(myMatrix.ndim))
myMatrix2 = np.array([[1,2,3],[3,4,5],[5,6,7]])
print(myMatrix2)
myMatrix3 = np.zeros((5,5)) #note that we pass a tuple, not a list
print(myMatrix3)
#instead of zeros we can allocate matrix with ones
myMatrix4 = np.ones((5,5))
print(myMatrix4)
#or whatever we want
myMatrix5 = np.full((5,5),9)
print(myMatrix5)
myMatrix5[2,2] = 0
print(myMatrix5)
ma = np.array([[1,2,3],[4,5,6]])
mb = np.array([[1,2,3],[4,5,6]])

# multiply each member of ma by respective member of mb
print(ma*mb)

# multiply each member of ma by 5
print(ma*5)
# reverse
print(np.array([1,2,3,4,5])[::-1])

# box selection
b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(b[1:3,1:3])

# take last row of matrix
print(b[-1,:])

# take last column of matrix
print(b[:,-1])
# flatten( convert matrix into array)
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a1 = a.ravel()
print("flatten( convert matrix into array)")
print(a1)

#reshape array back to matrix
a2 = a1.reshape(3,3)
print("reshape array back to matrix")
print(a2)

#transpose
print("transpose")
print(a2.T)

# get shape
print("get shape")
print(a2.shape)

# stacking
c = np.array([[1,2],[3,4]])
d = np.array([[-1,-2],[-3,-4]])

stackv = np.vstack((c,d))
stackh = np.hstack((c,d))

print("stack vertical")
print(stackv)
print("stack horizontal")
print(stackh)
# it's a python list of numbers
list1 = [1,2,3,4]
print(type(list1))

#make it a numpy array as we saw earlier
arr1 = np.array(list1)
print(type(arr1))

# make it a python list of numbers back, if needed
list2 = list(arr1)
print(type(list2))
someArray = np.array([1,2,3,4,5,6,7,8]) # create an array
anotherArray = someArray # thinking you are copying it into another(but you're mistaken)
print(someArray)
print(anotherArray)
someArray[1] = 999 # change the first
print(someArray)
print(anotherArray) # and see that second one changed too
someArray = np.array([1,2,3,4,5,6,7,8]) # create an array
anotherArray = someArray.copy() # copy it(properly)
print(someArray)
print(anotherArray)
someArray[1] = 999 # change the first
print(someArray)
print(anotherArray)# the second is not changed.