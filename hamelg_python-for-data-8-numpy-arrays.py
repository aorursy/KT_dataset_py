import numpy as np
my_list = [1, 2, 3, 4]             # Define a list



my_array = np.array(my_list)       # Pass the list to np.array()



type(my_array)                     # Check the object's type


second_list = [5, 6, 7, 8]



two_d_array = np.array([my_list, second_list])



print(two_d_array)


two_d_array.shape
two_d_array.size
two_d_array.dtype
# np.identity() to create a square 2d array with 1's across the diagonal



np.identity(n = 5)      # Size of the array
# np.eye() to create a 2d array with 1's across a specified diagonal



np.eye(N = 3,  # Number of rows

       M = 5,  # Number of columns

       k = 1)  # Index of the diagonal (main diagonal (0) is default)
# np.ones() to create an array filled with ones:



np.ones(shape= [2,4])
# np.zeros() to create an array filled with zeros:



np.zeros(shape= [4,6])
one_d_array = np.array([1,2,3,4,5,6])



one_d_array[3]        # Get the item at index 3
one_d_array[3:]       # Get a slice from index 3 to the end
one_d_array[::-1]     # Slice backwards to reverse the array
# Create a new 2d array

two_d_array = np.array([one_d_array, one_d_array + 6, one_d_array + 12])



print(two_d_array) 
# Get the element at row index 1, column index 4



two_d_array[1, 4]
# Slice elements starting at row 2, and column 5



two_d_array[1:, 4:]
# Reverse both dimensions (180 degree rotation)



two_d_array[::-1, ::-1]
np.reshape(a=two_d_array,        # Array to reshape

           newshape=(6,3))       # Dimensions of the new array
np.ravel(a=two_d_array,

         order='C')         # Use C-style unraveling (by rows)
np.ravel(a=two_d_array,

         order='F')         # Use Fortran-style unraveling (by columns)
two_d_array.flatten()
two_d_array.T
np.flipud(two_d_array)
np.fliplr(two_d_array)
np.rot90(two_d_array,

         k=1)             # Number of 90 degree rotations
np.roll(a= two_d_array,

        shift = 2,        # Shift elements 2 positions

        axis = 1)         # In each row
np.roll(a= two_d_array,

        shift = 2)
array_to_join = np.array([[10,20,30],[40,50,60],[70,80,90]])



np.concatenate( (two_d_array,array_to_join),  # Arrays to join

               axis=1)                        # Axis to join upon
two_d_array + 100    # Add 100 to each element
two_d_array - 100    # Subtract 100 from each element
two_d_array * 2      # Multiply each element by 2
two_d_array ** 2      # Square each element
two_d_array % 2       # Take modulus of each element 
small_array1 = np.array([[1,2],[3,4]])



small_array1 + small_array1
small_array1 - small_array1
small_array1 * small_array1
small_array1 ** small_array1
# Get the mean of all the elements in an array with np.mean()



np.mean(two_d_array)
# Provide an axis argument to get means across a dimension



np.mean(two_d_array,

        axis = 1)     # Get means of each row
# Get the standard deviation all the elements in an array with np.std()



np.std(two_d_array)
# Provide an axis argument to get standard deviations across a dimension



np.std(two_d_array,

        axis = 0)     # Get stdev for each column


# Sum the elements of an array across an axis with np.sum()



np.sum(two_d_array, 

       axis=1)        # Get the row sums
np.sum(two_d_array,

       axis=0)        # Get the column sums
# Take the log of each element in an array with np.log()



np.log(two_d_array)
# Take the square root of each element with np.sqrt()



np.sqrt(two_d_array)
# Take the vector dot product of row 0 and row 1



np.dot(two_d_array[0,0:],  # Slice row 0

       two_d_array[1,0:])  # Slice row 1
# Do a matrix multiply



np.dot(small_array1, small_array1)