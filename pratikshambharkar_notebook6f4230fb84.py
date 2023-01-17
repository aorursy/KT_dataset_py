!pip install jovian --upgrade -q
import jovian
jovian.commit(project='numpy-array-operations')
import numpy as np
# List of functions explained 
function1 = np.loadtxt("file.txt")  #loads a txt file
function2 = numpy.sum([1,2,3]) #Sums the array on the basis of axis
function3 = numpy.vstack() # stacks the array vertically
function4 = numpy.hstack() # stacks the array horizontly
function5 = numpy.reshape() # fits into given shape
import numpy as np
file = np.loadtxt("file.txt", dtype="str") #loading a data from txt in string format
print(file)
# Example 1 - working (change this)
data = np.loadtxt("file.txt", dtype={'names': ('gender', 'age', 'name'), 'formats':('S1', 'i4', 'U25')})
#loading a data from txt in object format
print(data)
data["gender"] #printing all genders
# Example 2 - working
data = np.loadtxt("file.txt", dtype="str", delimiter=" ") 
print(data)
# Example 3 - breaking (to illustrate when it breaks)
file = np.loadtxt("file.txt")
jovian.commit()
# Example 1 - working
arr = np.sum([1,2,3])
print(arr)
# Example 2 - working
arr = [
       [1,2,3],
       [1,2,3]
      ]
arr = np.sum(arr,axis=0)
print(arr)
# Example 3 - breaking (to illustrate when it breaks)
arr = [
    [1,2,3],
    [1,2],
    [1,2,3]
]

arr = np.sum(arr,axis=0)
print(arr)
jovian.commit()
# Example 1 - working
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])

arr = np.vstack((a,b))
print(arr)
# Example 2 - working
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
c = np.array([5, 6, 7])

arr = np.vstack((a,b,c))
print(arr)
# Example 3 - breaking (to illustrate when it breaks)
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])

arr = np.vstack(a,b)
print(arr)
jovian.commit()
# Example 1 - working
a = np.array((1,2,3))
b = np.array((2,3,4))
np.hstack((a,b))
# Example 2 - working
a = np.array((1,2,3))
b = np.array((2,3,4))
c = np.array((5,6,7))

np.hstack((a,b,c))
# Example 3 - breaking (to illustrate when it breaks)
a = np.array((1,2,3))
b = np.array((2,3,4))

np.hstack(a,b)
jovian.commit()
# Example 1 - working
a = np.arange(6)
a = a.reshape((3, 2))

a
# Example 2 - working
a = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [1,2,3]
]
a = np.array(a)
a = a.reshape((3,4))
a
# Example 3 - breaking (to illustrate when it breaks)
a = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [1,2,3]
]

a = a.reshape((3,4))
a
jovian.commit()
jovian.commit()
