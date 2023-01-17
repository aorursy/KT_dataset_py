import sys

import numpy as np

print("This cheatsheet was run on python {pv} and numpy version {nv}".format(pv=sys.version,nv=np.__version__))
#converting a simple list into numpy array - Ex - {1,2,3,4,5}

simple_list = [9,5,2,7,45]

numpy_arr = np.array(simple_list)

#1. Numpy array

print("1.  Numpy list : ",numpy_arr)



#2. Datatype of numpy array

print("2.  Datatype of numpy array : ",type(numpy_arr))



#3 Datatype of element in a numpy arr

print("3.  Datatype of elements in array : ",numpy_arr.dtype)



#4 Create a numpy array of floating values

print("4a. Numpy array of floating elements : ",np.array([1.1,2.2,3.5,6.7],dtype=float))

print("4b. Datatype of elements : ",np.array([1.1,2.2,3.5,6.7],dtype=float).dtype)



# Create a 2d numpy array

print("5.  2D numpy array : \n",np.array([[1,2],[3,4]]))
#1. Generate an array of zeros

print("1a. 1D Array of zeros : ",np.zeros(10)) #scaler input for 1d array

print("1b. 3x4 2D Array of zeros : \n",np.zeros((3,4))) #tuple input for nd array



#2. Generate an array of ones

print("2a. 1D Array of ones : ",np.ones(10,dtype=int))

print("2b. 3x4 2D Array of ones : \n",np.ones((3,4)))





#3. Generate a 1d array of uniformly incremented integers starting from x till y with step z(increment value for each element)

# z(step is optional if not specified,default value as 1)

print("3.  Array of integers starting from 1 till 12 with step 2",np.arange(1,12,2)) 



#4. Generate a 1d array of evenly spaced numbers over a specified interval

print("4.  Array of 4 evenly space numbers between 1 and 9",np.linspace(1,9,4)) # x,y,z : z number of values are generated between x and y 



#5. Generate an identity matrix

print("5.  4x4 Identity matrix \n",np.eye(4,dtype=int))



#6. Generate a 3x4 matrix with all elements as 7

print("6.  3x4 matrix of 7s\n",np.full((3,4),7))



#7. Generate a random 1d array of integers where each elements should be between 1 to 10

# input to method (x,y,z) z number of elements are created with each randomly generated integer lying between x and y

print("7.  Array of 10 random integers ranging from 1-10",np.random.randint(1,10,10))



#8. Generate a uniformly distributed 2x4 matrix

print("8.  3x4 uniformly distributed random matrix\n",np.random.rand(3,4))

#1. Shape of the array

simple_arr = np.array([1,2,3,4,5])

two_d_array = np.array([[1,2],[3,4]])

print("1a. 1D array shape : ",simple_arr.shape)

print("1b. 2D array shape : ",two_d_array.shape)



#2. length of the array

print("2.  Length of an array : ",len(simple_arr))



#3. Number of dimension an array has

print("3.  Number of dimensions in an array : ",two_d_array.ndim)



#4. Number elements in the array

print("4.  Number of elements in an array : ",two_d_array.size)



#5. Datatype of elements in an array

print("5.  Datatype of the elements in an array : ",simple_arr.dtype.name)



#6 Convert the datatype of elements in an array

new_arr = simple_arr.astype(str)

print("6b. Converted array : ",new_arr)

print("6b. Converted array's element datatype : ",new_arr.dtype.name)
#index numpy array starts from 0

numpy_arr = np.array([10,20,30,40,50,60,70,80,90])

print("1. Array : ",numpy_arr)



#Access and print the element at 2th index in numpy array numpy_arr

print("2. Element at 0th index : ",numpy_arr[0])





#3. Access and print the 2nd element from last. Note : Numpy has reverse/negative index on every array 

#      that helps in traversing in resever direction or access elements from last to first.

print("3. ",numpy_arr[-2])



#The basic slice syntax is i:j:k where i is the starting index, j is the stopping index, and k is the step



#1. Printing all the element from 0th index till 5th index(not included)

print("1. ",numpy_arr[0:5:1])



#2. Another way of printing above 

print("2. ",numpy_arr[:5])



#3. Print all elements from 2th index till the last index(including)

print("3. ",numpy_arr[2:])



#4. Print all elements form 0th index till the last skipping one element

print("4. ",numpy_arr[::2])



#5. Print all elements but last 2 elements

print("5. ",numpy_arr[:-2])



#6. Print only last 2 elements

print("6. ",numpy_arr[-2:])

numpy_arr = np.array([[5,7,13,4],[41,2,23,67],[3,34,9,8],[11,56,0,16]])

numpy_arr
#1. pick the element at index 2,1. 2D array takes index as [n,m] or [n][m] where n is the row index and m the column index

print("1a. Element at 2,1 : ",numpy_arr[2][1])

print("1b. Element at 2,1 : ",numpy_arr[2,1])



#2. Fetch the first row of the matrix

print("2. First row : ",numpy_arr[0])





#3. Fetch the 2nd column of the matrix

print("3. First Column : ",numpy_arr[:,1]) # slice before comma denotes we want all the element from row, 

                                        #1 after comma denotes column position as 1

    

#4. Fetch all column but first

print("4. \n",numpy_arr[:,1:])
numpy_arr[:3,1]
numpy_arr = np.array([1,2,3,4,5])

print("1.  Base array : ",numpy_arr)



#2. Adding new element 6 to the array and remove the element at 2nd index

numpy_arr = np.append(numpy_arr,6)

print("2a. Appending 6 to numpy_arr : ",numpy_arr)

print("2b. Deleting element at index 2 : ",np.delete(numpy_arr,2))



#3. Adding a new element at a particular position

print("3.  Inserting 22 at index 1 : ",np.insert(numpy_arr,1,22))



#4. Creating a 2d array

two_d_array = [[1,2,3],

              [4,5,6],

              [7,8,9]]

two_d_np_array = np.array(two_d_array)

print("4.  2D array \n",two_d_np_array)



#5. Add a row [-1,-2,-3] and remove the 0th row from the 2d array two_d_np_array

two_d_np_array = np.vstack((two_d_np_array,[-1,-2,-3]))

print("5a. Adding a new new row : \n",two_d_np_array)



two_d_np_array = np.delete(two_d_np_array,0,axis=0)

print("5b. Deleting a row : \n",two_d_np_array)



#6. Add a clumn [10,20,30] to two_d_np_array

two_d_np_array = np.column_stack((two_d_np_array,[10,20,30]))

print("6. Adding a new column : \n",two_d_np_array)



#7. Append 2 2d matrix column wise

a = [[1,2,3],[4,5,6],[7,8,9]]

b = [[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]]

print("7. Appending 2 2D numpy arrays : \n",np.append(a,b,axis=0))





#8. convert a 2d matrix to a flat 1dmatrix/array

a = np.array([[1,2,3],[4,5,6],[7,8,9]])

print("8. flattening an nD array to 1d array : ",a.flatten())



#9. Reshape a 4x2 matrix to 2x4 matrix

a = np.array([[1,2],[3,4],[5,6],[7,8]])

print("9a. original shape",a.shape)

print("9b. New shape ",a.reshape((2,4)).shape)

print("9c. value \n",a.reshape((2,4)))



#10. Transpose a matrix

print("10. Transpose : \n",np.transpose(a))





#11. Split an matrix horizontally

print("11. Horizontal split : \n",np.hsplit(a,1))



#12. Split an matrix vertically

print("12. Vertical split : \n",np.vsplit(a,1))
#1. Add all the elements of an numpy array with 2, Note, Numpy allows scaler operation on each elements

numpy_arr = np.array([4,12,64,8,12,128])

print("1a. Original array : ",numpy_arr)

print("1b. Adding 2 to all elements of the array : ",numpy_arr+2)



#2. Divide all the elements of array by 4

print("2. Divding all elements with 4 : ",numpy_arr/4)



#3. Perform above action using lamba

div_custom = lambda x: x/4

print("3. Apply a lamda func to all elements : ",div_custom(numpy_arr))



#4. add each elements of one array with corresponsing elements of another

a = np.array([1,2,3])

b = np.array([4,16,64])

print("4. Adding 2 arrays(element wise) : ",np.add(a,b))



#5. Substract each elements of one array from corresponsing elements of another

print("5. Substracting one array from another (Element wise) : ",np.subtract(b,a))



#6. Divide each elements of one array by corresponsing elements of another

print("6. Divide one array with another(Element wise) : ",np.divide(b,a))



#7. Multiply each elements of one array by corresponsing elements of another

print("7. Muliply one array with another(Element wise) : ",np.multiply(b,a))



#8. Square root of each element in an array

print("8a. Square root of each element :",np.sqrt(b))

print("8b. Generic way of square root : ",b**.5)



#9. Check if elements of one array is greater than corresponsing elements in another array

print("9. Element wise comparision : ",b>a)
a = np.array([10,2,13,44,9,6])



#1. Getting max element in the array

print("1. Element with highest numerical value : ",a.max())



#2. Getting position of max element in the array

print("2. Position of highest number : ",a.argmax())



#3. Getting max element in the array

print("3. Index with least numerical value : ",a.min())



#4. Getting position of max element in the array

print("4. Index of least number : ",a.argmin())



#5. Print the mean of all elements in the array

print("5. Mean of all the element : ",a.mean())



#6. Print the variance of all elements in the array

print("6. Variance of all the element : ",a.var())



#7. Print the size of the array

print("7. Size of the array : ",a.size)



#8. Sum of all the elements

print("8. Sum of all elements : ",a.sum())
numpy_arr = np.array([12,33,2,56,4,32,7])

print(numpy_arr)

#1. Create a new array with by fetching all elements in numpy_arr that are greater than 10



print("1. condition on each element of array : ",numpy_arr>10)



new_arr = numpy_arr[numpy_arr>10]

print("1. Passing the above boolean array to array itself fetches all elements that corresponds to True : ",new_arr)
numpy_arr = np.arange(6)

print("1. Original Array : \n",numpy_arr)



#2. shallow copy

arr_view = numpy_arr.view()

print("2. Shallow copy : \n",arr_view)





#3. copy/Deep copy

arr_copy = numpy_arr.copy()

print("3. Deep copy : \n",arr_copy)