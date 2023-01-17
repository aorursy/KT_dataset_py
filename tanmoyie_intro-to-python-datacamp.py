var_list = ['py', 2, '3rd', 'last_element']

# Subsetting list

# Calling 1st element

print(var_list[0]) # Python is a zero-indexed programming language

# Calling last element

print(var_list[-1])
print('1-3: ', var_list[1:3]) # calling 2nd, 3rd element. But, many are expecting the 4th element to be called as well.
print(var_list[1:]) # taking 2nd element and onwards; again inclusive

print(var_list[:2]) # taking until 2nd element (not 3rd element); again exclusive
import numpy as np

print(np.array([2, False, 'Tanmoy'])) # When it comes to mixed datatype, Numpy convert everything into str unlike Python LIST that preserves datatype

print([2, False, 'Das'])

py_list = [2,4,5]

numpy_array = np.array([2, 4 ,6])

sum_np = numpy_array + numpy_array

print('Summing Python Lists: ', (py_list + py_list)) # Concatinating them

print('Summing numpy arrays: ', sum_np)
# More contents will be added shortly thereafter.