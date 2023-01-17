# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

badCompareChance = 0

# Any results you write to the current directory are saved as output.
def badCompare(r):

    if rnd.choice(range(1000)) < (1000 * r):

        return True

    else:

        return False
def partition(arr,low,high, r): 

    i = ( low-1 )         # index of smaller element 

    pivot = arr[high]     # pivot 

  

    for j in range(low , high): 

  

        # If current element is smaller than or 

        # equal to pivot 

        if   arr[j] <= pivot: 

            if (not badCompare(r)):

                # increment index of smaller element 

                i = i+1 

                arr[i],arr[j] = arr[j],arr[i] 

  

    arr[i+1],arr[high] = arr[high],arr[i+1] 

    return ( i+1 ) 

  

# The main function that implements QuickSort 

# arr[] --> Array to be sorted, 

# low  --> Starting index, 

# high  --> Ending index 



def quickSort(arr, r):

    quicksort(arr, 0, len(arr) - 1, r)



# Function to do Quick sort 

def quicksort(arr,low,high, r): 

    if low < high: 

  

        # pi is partitioning index, arr[p] is now 

        # at right place 

        pi = partition(arr,low,high, r) 

  

        # Separately sort elements before 

        # partition and after partition 

        quicksort(arr, low, pi-1, r) 

        quicksort(arr, pi+1, high, r) 
def mergeSort(arr, r): 

    if len(arr) >1: 

        mid = len(arr)//2 #Finding the mid of the array 

        L = arr[:mid] # Dividing the array elements  

        R = arr[mid:] # into 2 halves 

  

        mergeSort(L, r) # Sorting the first half 

        mergeSort(R, r) # Sorting the second half 

  

        i = j = k = 0

          

        # Copy data to temp arrays L[] and R[] 

        while i < len(L) and j < len(R): 

            if L[i] < R[j] and not badCompare(r): 

                arr[k] = L[i] 

                i+=1

            else: 

                arr[k] = R[j] 

                j+=1

            k+=1

          

        # Checking if any element was left 

        while i < len(L): 

            arr[k] = L[i] 

            i+=1

            k+=1

          

        while j < len(R): 

            arr[k] = R[j] 

            j+=1

            k+=1
def heapify(arr, n, i, rate): 

    largest = i  # Initialize largest as root 

    l = 2 * i + 1     # left = 2*i + 1 

    r = 2 * i + 2     # right = 2*i + 2 

  

    # See if left child of root exists and is 

    # greater than root 

    if (l < n) and (arr[i] < arr[l]) and not badCompare(rate): 

        largest = l 

  

    # See if right child of root exists and is 

    # greater than root 

    if (r < n) and (arr[largest] < arr[r]) and not badCompare(rate): 

        largest = r 

  

    # Change root, if needed 

    if (largest != i) and not badCompare(rate):

        arr[i],arr[largest] = arr[largest],arr[i]  # swap 

  

        # Heapify the root. 

        heapify(arr, n, largest, rate)

  

# The main function to sort an array of given size 

def heapSort(arr, rate): 

    n = len(arr) 

  

    # Build a maxheap. 

    for i in range(n, -1, -1): 

        heapify(arr, n, i, rate) 

  

    # One by one extract elements 

    for i in range(n-1, 0, -1): 

        arr[i], arr[0] = arr[0], arr[i]   # swap 

        heapify(arr, i, 0, rate) 
def insertionSort(arr, r): 

  

    # Traverse through 1 to len(arr) 

    for i in range(1, len(arr)): 

  

        key = arr[i] 

  

        # Move elements of arr[0..i-1], that are 

        # greater than key, to one position ahead 

        # of their current position 

        j = i-1

        while j >=0 and key < arr[j] and not badCompare(r): 

                arr[j+1] = arr[j] 

                j -= 1

        arr[j+1] = key 
def bubbleSort(arr, r): 

    n = len(arr) 

  

    # Traverse through all array elements 

    for i in range(n): 

  

        # Last i elements are already in place 

        for j in range(0, n-i-1): 

  

            # traverse the array from 0 to n-i-1 

            # Swap if the element found is greater 

            # than the next element 

            if arr[j] > arr[j+1] and not badCompare(r): 

                arr[j], arr[j+1] = arr[j+1], arr[j] 
def selectionSort(arr, r):

    for i in range(len(arr)): 

      

        # Find the minimum element in remaining  

        # unsorted array 

        min_idx = i 

        for j in range(i+1, len(arr)): 

            if arr[min_idx] > arr[j] and not badCompare(r): 

                min_idx = j 



        # Swap the found minimum element with  

        # the first element         

        arr[i], arr[min_idx] = arr[min_idx], arr[i]
n = 1000

def testSortZeroBadCompare(sort, sortName):

    arr = list(range(n))

    rnd.shuffle(arr)

    sort(arr, 0)

    print(sortName + " sorted arr correctly?", arr == list(range(n)))
testSortZeroBadCompare(quickSort, "quickSort")

testSortZeroBadCompare(mergeSort, "mergeSort")

testSortZeroBadCompare(heapSort, "heapSort")

testSortZeroBadCompare(insertionSort, "insertionSort")

testSortZeroBadCompare(selectionSort, "selectionSort")

testSortZeroBadCompare(bubbleSort, "bubbleSort")
def getMistakes(arr):

    mistakes = [i for i in arr[1:-1] if arr[arr.index(i) - 1] > i or arr[arr.index(i) + 1] < i]

    if arr[1] < arr[0]:

        mistakes.append(arr[0])

    if (arr[n-2] > arr[n-1]):

        mistakes.append(arr[n-1])

    

    return mistakes
nRates = 75

jump = 0.1/nRates

currRate = 0.001

rates = []

for i in range(nRates):

    rates.append(currRate)

    currRate += jump



def testSort(sort, sortName):

    

    n = 2000

    results = []

    seed = 6572348997656764534

    for r in rates:

        for trial in range(30):

            arr = list(range(n))

            rnd.seed(seed)

            rnd.shuffle(arr)

            seed += 1

            rnd.seed(seed)

            sort(arr, r)

            mistakes = getMistakes(arr)

            results.append([r, len(mistakes) / n])

    print(sortName, "Done")

    return results
def save(arr, name):

    f = open(name + ".txt", "w")

    for i in arr:

        f.write(" ".join(map(str,i)) + "\n")



def restore(name):

    arr = []

    f = open("../input/75rates/" + name + ".txt", "r")

    for i in f:

        arr.append([float(i.split()[0]), float(i.split()[1])])

    return arr
ranBefore = True

if not ranBefore:

    QS = testSort(quickSort, "QuickSort")

    MS = testSort(mergeSort, "MergeSort")

    HS = testSort(heapSort, "HeapSort")

    IS = testSort(insertionSort, "InsertionSort")

    SS = testSort(selectionSort, "SelectionSort")

    BS = testSort(bubbleSort, "BubbleSort")

    

    save(QS, "QuickSort")

    save(MS, "MergeSort")

    save(HS, "HeapSort")

    save(IS, "InsertionSort")

    save(SS, "SelectionSort")

    save(BS, "BubbleSort")

    

else:

    QS = restore("QuickSort")

    MS = restore("MergeSort")

    HS = restore("HeapSort")

    IS = restore("InsertionSort")

    SS = restore("SelectionSort")

    BS = restore("BubbleSort")
x = [i[0] for i in QS]

y = [i[1] for i in QS]



fig = px.scatter(x=x, y=y, color=y, title="QuickSort Raw")

fig.show()



x = [np.log(i[0]) for i in QS]

y = [np.log(i[1]) for i in QS]



fig = px.scatter(x=x, y=y, color=y, title="QuickSort Ln")

fig.show()
x = [i[0] for i in MS]

y = [i[1] for i in MS]



fig = px.scatter(x=x, y=y, color=y, title="MergeSort Raw")

fig.show()



x = [np.log(i[0]) for i in MS]

y = [np.log(i[1]) for i in MS]



fig = px.scatter(x=x, y=y, color=y, title="MergeSort Ln")

fig.show()
x = [i[0] for i in HS]

y = [i[1] for i in HS]



fig = px.scatter(x=x, y=y, color=y, title="HeapSort Raw")

fig.show()



x = [np.log(i[0]) for i in HS]

y = [np.log(i[1]) for i in HS]



fig = px.scatter(x=x, y=y, color=y, title="HeapSort Ln")

fig.show()
x = [i[0] for i in IS]

y = [i[1] for i in IS]



fig = px.scatter(x=x, y=y, color=y, title="InsertionSort Raw")

fig.show()



x = [np.log(i[0]) for i in IS]

y = [np.log(i[1]) for i in IS]



fig = px.scatter(x=x, y=y, color=y, title="InsertionSort Ln")

fig.show()
x = [i[0] for i in SS]

y = [i[1] for i in SS]



fig = px.scatter(x=x, y=y, color=y, title="SelectionSort Raw")

fig.show()



x = [np.log(i[0]) for i in SS]

y = [np.log(i[1]) for i in SS]



fig = px.scatter(x=x, y=y, color=y, title="SelectionSort Ln")

fig.show()
x = [i[0] for i in BS]

y = [i[1] for i in BS]



fig = px.scatter(x=x, y=y, color=y, title="BubbleSort Raw")

fig.show()



x = [np.log(i[0]) for i in BS]

y = [np.log(i[1]) for i in BS]



fig = px.scatter(x=x, y=y, color=y, title="BubbleSort Ln")

fig.show()




QS = [("quickSort" + str(i[0]), i[1]) for i in QS]

MS = [("mergeSort" + str(i[0]), i[1]) for i in MS]

HS = [("heapSort" + str(i[0]), i[1]) for i in HS]

IS = [("insertionSort" + str(i[0]), i[1]) for i in IS]

SS = [("selectionSort" + str(i[0]), i[1]) for i in SS]

BS = [("bubbleSort" + str(i[0]), i[1]) for i in BS]



x = [i[0] for i in QS]

y = [i[1] for i in QS]



x += [i[0] for i in MS]

y += [i[1] for i in MS]



x += [i[0] for i in HS]

y += [i[1] for i in HS]



x += [i[0] for i in SS]

y += [i[1] for i in SS]



x += [i[0] for i in IS]

y += [i[1] for i in IS]



x += [i[0] for i in BS]

y += [i[1] for i in BS]



fig = px.scatter(x=x, y=y, color=y)

fig.show()