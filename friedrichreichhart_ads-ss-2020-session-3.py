# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
range(5)
for i in range(5):
    print(i)
range(1,6)
array = [5, 3, 9, 11, 20, 3]
print(array)
len(array)
for i in range(len(array)):
    print(i)
for i in range(len(array)):
    print(array[i])
for i in range(len(array)-1):
    print(array[i])
    print(array[i+1])
    print("---")
for i in range(len(array)-1): #outer loop (index i)
    print("iteration #", (i+1))
    for j in range(len(array)-1-i): #inner/nested loop (index j)
        print(array[j])
        print(array[j+1])
        print("---")        
print("unsorted")
print(array)

for i in range(len(array)-1): #outer loop (index i)
    print("iteration #", (i+1))
    for j in range(len(array)-1-i): #inner/nested loop (index j)
        if (array[j] > array[j+1]):
            helper = array[j]
            array[j] = array[j+1]
            array[j+1] = helper

print("sorted")
print(array)
array = [3,44,38,5,47,15,36,26,27,2,46,4,19,50,48]
print("unsorted")
print(array)

for i in range(len(array)-1): #outer loop (index i)
    print("iteration #", (i+1))
    for j in range(len(array)-1-i): #inner/nested loop (index j)
        if (array[j] > array[j+1]):
            helper = array[j]
            array[j] = array[j+1]
            array[j+1] = helper
    print(array)
print("sorted")
print(array)
array = [3,44,38,5,47,15,36,26,27,2,46,4,19,50,48]
print("unsorted")
print(array)

comparison_count = 0
swap_count = 0

for i in range(len(array)-1): #outer loop (index i)
    print("iteration #", (i+1))
    for j in range(len(array)-1-i): #inner/nested loop (index j)
        # cpmparison count
        comparison_count = comparison_count + 1
        if (array[j] > array[j+1]):
            helper = array[j]
            array[j] = array[j+1]
            array[j+1] = helper
            # swap count
            swap_count = swap_count +1
            
    print(array)
print("sorted")
print(array)
print("---------")
print("comparison_count: ", comparison_count)
print("swap_count: ", swap_count)
print("unsorted")
print(array)

comparison_count = 0
swap_count = 0

for i in range(len(array)-1): #outer loop (index i)
    print("iteration #", (i+1))
    for j in range(len(array)-1-i): #inner/nested loop (index j)
        # cpmparison count
        comparison_count = comparison_count + 1
        if (array[j] > array[j+1]):
            helper = array[j]
            array[j] = array[j+1]
            array[j+1] = helper
            # swap count
            swap_count = swap_count +1
            
    print(array)
print("sorted")
print(array)
print("---------")
print("comparison_count: ", comparison_count)
print("swap_count: ", swap_count)
print("unsorted")
print(array)

comparison_count = 0
swap_count = 0

for i in range(len(array)-1): #outer loop (index i)
    print("iteration #", (i+1))
    for j in range(len(array)-1-i): #inner/nested loop (index j)
        # cpmparison count
        comparison_count = comparison_count + 1
        if (array[j] < array[j+1]):
            helper = array[j]
            array[j] = array[j+1]
            array[j+1] = helper
            # swap count
            swap_count = swap_count +1
            
    print(array)
print("sorted")
print(array)
print("---------")
print("comparison_count: ", comparison_count)
print("swap_count: ", swap_count)
array = [5, 3, 9, 11, 20, 3]
print(array)
for i in range(len(array)):
    min_index = i
    for j in range(i+1, len(array)):
        if (array[min_index] > array[j]):
            min_index = j
    helper = array[min_index]
    array[min_index] = array[i]
    array[i] = helper
    print(array)

print(array)
array = [3,44,38,5,47,15,36,26,27,2,46,4,19,50,48]

print("unsorted")
print(array)

comparison_count = 0
swap_count = 0

for i in range(len(array)): # outer loop
    print("iteration #", (i+1))
    min_index = i
    for j in range(i+1, len(array)): # nested loop
        comparison_count = comparison_count + 1
        if (array[min_index] > array[j]):
            min_index = j
    # swap after iteration round
    # swap count
    if (min_index != i):
        swap_count = swap_count +1
        helper = array[min_index]
        array[min_index] = array[i]
        array[i] = helper
    
    print(array)
    
print("sorted")
print(array)
print("---------")
print("comparison_count: ", comparison_count)
print("swap_count: ", swap_count)
print("unsorted")
print(array)

comparison_count = 0
swap_count = 0

for i in range(len(array)): # outer loop
    print("iteration #", (i+1))
    min_index = i
    for j in range(i+1, len(array)): # nested loop
        comparison_count = comparison_count + 1
        if (array[min_index] > array[j]):
            min_index = j
    # swap after iteration round
    # swap count
    if (min_index != i):
        swap_count = swap_count +1
        helper = array[min_index]
        array[min_index] = array[i]
        array[i] = helper
    
    print(array)
    
print("sorted")
print(array)
print("---------")
print("comparison_count: ", comparison_count)
print("swap_count: ", swap_count)
print("unsorted")
print(array)

comparison_count = 0
swap_count = 0

for i in range(len(array)): # outer loop
    print("iteration #", (i+1))
    min_index = i
    for j in range(i+1, len(array)): # nested loop
        comparison_count = comparison_count + 1
        if (array[min_index] < array[j]):
            min_index = j
    # swap after iteration round
    # swap count
    if (min_index != i):
        swap_count = swap_count +1
        helper = array[min_index]
        array[min_index] = array[i]
        array[i] = helper
    
    print(array)
    
print("sorted descending")
print(array)
print("---------")
print("comparison_count: ", comparison_count)
print("swap_count: ", swap_count)
for i in range(1,5):
    print(i)
store = array[i]
j = i-1
array = [3,44,38,5,47,15,36,26,27,2,46,4,19,50,48]

print("unsorted")
print(array)

for i in range(1, len(array)):
    store = array[i]
    j = i-1
    while j >= 0 and store < array[j]:
        array[j+1] = array[j]
        j -= 1
    array[j+1] = store

print("sorted")
print(array)
array = [3,44,38,5,47,15,36,26,27,2,46,4,19,50,48]

comparison_count = 0
swap_count = 0

print("unsorted")
print(array)

for i in range(1, len(array)):
    store = array[i]
    j = i-1
    while j >= 0 and store < array[j]:
        array[j+1] = array[j]
        j -= 1
        comparison_count += 1
        swap_count += 1
        
    array[j+1] = store
    swap_count += 1 
    # end of iteration round
    print("Iteration Round #", i)
    print(array)

print("sorted")
print(array)

print("---------")
print("comparison_count: ", comparison_count)
print("swap_count: ", swap_count)
comparison_count = 0
swap_count = 0

print("unsorted")
print(array)

for i in range(1, len(array)):
    store = array[i]
    j = i-1
    while j >= 0 and store < array[j]:
        array[j+1] = array[j]
        j -= 1
        comparison_count += 1
        swap_count += 1
        
    array[j+1] = store
    swap_count += 1 
    # end of iteration round
    print("Iteration Round #", i)
    print(array)

print("sorted")
print(array)

print("---------")
print("comparison_count: ", comparison_count)
print("swap_count: ", swap_count)