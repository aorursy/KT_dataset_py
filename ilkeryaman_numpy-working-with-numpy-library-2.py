import numpy as np # linear algebra
arr = np.arange(1, 10)

arr
arr[:3] = 25 # set value of first 3 elements of array to 25.

print(arr)
arr = np.arange(1, 10)

arr2 = arr

arr2[:3] = 100 # set value of first 3 elements of array to 100.
print(arr)

print(arr2)

# Since objects are referenced in memory, both array are changed.
arr = np.arange(1, 10)

arr2 = arr.copy()

arr2[:3] = 100
print(arr)

print(arr2)
new_array = np.arange(1, 21).reshape(5, 4)

new_array
new_array[:, 0:2] # First argument says that get all elements. Second argument says that get elements at index of 0 to 2 for each element.
new_array[:3, :3] # First argument says that get first 3 elements. Second argument says that get elements at first 3 index for each element.
new_array[0:2, :] # First argument says that get elements from 0 to 2. Second argument says that get all elements for each element.
new_array[:2] # Get first 2 elements.
arr
arr > 3
boolean_array = arr > 3
arr[boolean_array]
arr[arr > 3]