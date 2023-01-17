import numpy as np
arr = np.arange(0, 11)
arr
arr[0]
arr[1:5]
arr[:5]
arr[3:]
arr
arr[0:5] = 100

arr
arr = np.arange(0, 11)
arr
slice_of_arr = arr[0:6]
slice_of_arr
slice_of_arr[:] = 99

slice_of_arr
arr
arr_copy = arr.copy()
arr
arr_copy[:] = 200
arr_copy
arr
# mat[row, column]

# mat[row][col]



mat = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
mat
mat[0]
mat[0][0]
mat[0, 0]
mat[:2, ]
mat[:2, 1:]
mat[1:,:2]
# Conditional Selection

arr = np.arange(1, 11)

arr
arr > 4
bool_arr = arr > 4
arr[bool_arr]
arr[arr > 4]
arr[arr <= 9]