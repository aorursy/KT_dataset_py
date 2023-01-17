import numpy as np
ages = [10,50,21,78,85,98,52,45,36,25,11,8,5,79]

print(ages)

type(ages) #just to show you that this is a list :)
ages_array = np.array(ages)

print(ages_array)

type(ages_array) #just to show you that this is now a numpy array!
size = ages_array.size

shape = ages_array.shape

print(f'Size = {size} \nType of the above output = {type(size)} \nShape = {shape} \nType of the above output = {type(shape)} ')
# size can also be found with 'len' method which gives output in integer

size_len = len(ages_array)

print(size_len)
ages1 = [10,50,21,78,85,98,52,45,36,25,11,8,5,79]

ages2 = [10,50,21,78,85,98,52,45,36,25,11,8,5,79]

both = ages1 + ages2

both = np.array(both) 

reshaped = both.reshape((2,14))

print(reshaped) # now the array is 2D with each dimension having 14 elements
# indexing in the normal numpy array

print(f'The normal numpy array we created = {ages_array}')

print(f'Normal numpy array 5th index = {ages_array[4]}')
# indexing in the reshaped numpy array

print(f'The normal numpy array we created = {reshaped}')

print(f'Normal numpy array 5th index = {reshaped[1,4]}')
# general form of the syntax : arr[start:stop:step]

print(f'The array we are working on : \n{reshaped}')

print(f'The first three elements from the first row = {reshaped[0,0:3]}')

# alternative of the above code

print(f'The first three elements from the first row [alternative] = {reshaped[0,:3]}') #since we are starting from index 0 itself, we can omit the 0

print(f'The entire fourth column of the 2D array = {reshaped[:,3]}')
test = reshaped

print(f'The array initially: \n{test}')

print(f'Initial Value of 1st array 4th column {test[0,3]}')

test[0,3] = 1234

print(f'New Value of 1st array 4th column {test[0,3]}')

print(f'The array now: \n{test}')
print(f'The array initially: \n{test}')

#here is the fun part!

test[0,:] = 5678

print(f'The array now: \n{test}')
print(f'The array initially: \n{test}')

test[:5, :5] = 1010

print(f'The array now: \n{test}')
#Join a sequence of arrays along a new axis.

a = np.array([1, 2, 3])

b = np.array([2, 3, 4])

print(f'Basic Stacking : \n {np.stack((a, b))}')

print(f'Axis = -1 : \n {np.stack((a, b), axis = -1)}')

print(f'Axis = -2 : \n {np.stack((a, b), axis = -2)}')

print(f'Axis = -3 : \nGives error : axis -3 is out of bounds for array of dimension 2')
#hstack = Stack arrays in sequence horizontally (column wise).

a = np.array((1,2,3))

b = np.array((2,3,4))

print (f'hstack for arrays {a} and {b} is \n{np.hstack((a,b))}')

a = np.array([[1],[2],[3]])

b = np.array([[2],[3],[4]])

print (f'\nhstack for arrays {a} and {b} is \n{np.hstack((a,b))}')
#vstack Stack arrays in sequence vertically (row wise)

a = np.array((1,2,3))

b = np.array((2,3,4))

print (f'vstack for arrays {a} and {b} is \n{np.vstack((a,b))}')

a = np.array([[1],[2],[3]])

b = np.array([[2],[3],[4]])

print (f'\nvstack for arrays {a} and {b} is \n{np.vstack((a,b))}')
# dstack Stack arrays in sequence depth wise (along third axis).

a = np.array((1,2,3))

b = np.array((2,3,4))

print (f'dstack for arrays {a} and {b} is \n{np.dstack((a,b))}')

a = np.array([[1],[2],[3]])

b = np.array([[2],[3],[4]])

print (f'\ndstack for arrays {a} and {b} is \n{np.dstack((a,b))}')
#Join a sequence of arrays along an existing axis.

a = np.array([[1, 2], [3, 4]])

b = np.array([[5, 6]])

c = np.concatenate((a, b), axis=0)

print(f'Array a = {a}\nArray b = {b}\nConcatenated Array c = \n{c}')
# Same as above, this time axis = none

a = np.array([[1, 2], [3, 4]])

b = np.array([[5, 6]])

c = np.concatenate((a, b), axis=None)

print(f'Array a = {a}\nArray b = {b}\nConcatenated Array c = \n{c}')
# This one looks cool! axis = 1

a = np.array([[1, 2], [3, 4]])

b = np.array([[5, 6]])

c = np.concatenate((a, b.T), axis=1)

print(f'Array a = {a}\nArray b = {b}\nConcatenated Array c = \n{c}')
a = np.array((1,2,3))

print (f'Adding 2 to each array element : {a+2}')

print (f'multipying each array element with 2 : {a*2}')

# and they lived happily ever after (I mean we can go on like this!)
# sum of all elements in the array

a = np.array((1,2,3))

print(f'Sum of all elements in the array {a} is {a.sum()}')
# minimum vaule

a = np.array((1,2,3))

print(f'Minimum of all elements in the array {a} is {a.min()}')
# maximum vaule

a = np.array((1,2,3))

print(f'Maximum of all elements in the array {a} is {a.max()}')
# mean of all elements in an array

a = np.array((1,2,3,2,4,70005,6))

print(f'Mean of all elements in the array {a} is {a.mean()}')