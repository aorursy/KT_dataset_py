!pip install jovian --upgrade -q
import jovian
jovian.commit(project='numpy-array-operations')
import numpy as np
# List of functions explained 
function1 = np.sort
function2 = np.argsort
function3 = np.sort_complex
function4 = np.lexsort
function5 = np.partition
# Example 1 - working (change this)
arr1 = [[15, 18], 
        [30, 21]]
sorted_arr = np.sort(arr1, axis = 0)
print(sorted_arr)
# Example 2 - working
arr1 = [[15, 18], 
        [30, 21]]
sorted_arr = np.sort(arr1, axis = None)
print(sorted_arr)
# Example 3 - breaking (to illustrate when it breaks)
arr1 = [[15, 18], 
        [30, 21]]

sorted_arr = np.sort(arr1, axis = 3)
print(sorted_arr)
jovian.commit()
# Example 1 - working
arr = np.array([ 2, 0,  1, 5, 4, 2, 9])
argsorted_arr = np.argsort(arr, axis = 0)

print(argsorted_arr)
# Example 2 - working
arr = np.array([[ 2, 0, 1], [ 5, 4, 3]])
argsorted_arr = np.argsort(arr, kind = 'mergesort', axis = 1)

print(argsorted_arr)
# Example 3 - breaking (to illustrate when it breaks)
arr = np.array([ 2, 0,  1, 5, 4, 2, 9])
argsorted_arr = np.argsort(arr, order = '1')

print(argsorted_arr)
jovian.commit()
# Example 1 - working
arr1 = [9,8,1,2,4,4] 
arr2 = [4,1,0,4,2,1] 
lexsorted_arr = np.lexsort((arr2, arr1))

print(lexsorted_arr)
# Example 2 - working
surnames =    ('Thanvi',    'Mathur', 'Jain')
firstnames = ('Anirudhi', 'Animesh', 'Sneha')
lexsorted_arr = np.lexsort((firstnames, surnames))

print(lexsorted_arr)
# Example 3 - breaking (to illustrate when it breaks)
arr1 = [9,8,1,4] 
arr2 = [4,2,1] 
lexsorted_arr = np.lexsort((arr2, arr1))

print(lexsorted_arr)
jovian.commit()
# Example 1 - working
arr = np.array([9,6,1,0,4])
complexsort_arr = np.sort_complex(arr)

print(complexsort_arr)
# Example 2 - working
arr = np.array([3 + 4j, 5 - 1j, 9 - 3j, 2 - 2j, 1 + 3j])
complexsort_arr = np.sort_complex(arr)

print(complexsort_arr)
# Example 3 - breaking (to illustrate when it breaks)
arr = np.array([[3 + 4j, 5 - 1j],[ 9 - 3j, 2 - 2j, 1 + 3j]])
complexsort_arr = np.sort_complex(arr)

print(complexsort_arr)
jovian.commit()
# Example 1 - working
arr = np.array([4, 5, 3, 4, 2, 6, 1,])
partitioned_arr = np.partition(arr, 4)

print(partitioned_arr)
# Example 2 - working
arr = np.array([ 2, 0, 1, 5, 4, 9, 3]) 
partitioned_arr = np.partition(arr, (0, 3)) 

print(partitioned_arr)
# Example 3 - breaking (to illustrate when it breaks)
arr = np.array([ 2, 0, 1, 5, 4, 9, 3]) 
partitioned_arr = np.partition(arr) 

print(partitioned_arr)
jovian.commit()
jovian.commit()