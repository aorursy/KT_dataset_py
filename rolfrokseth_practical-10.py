#Merge sort

from bitstring import BitArray

from time import time

import matplotlib.pyplot as plt



timePlot = []

elePlot = []

t1 = time()



def mergeSort(alist):

    #print("Splitting ",alist)

    if len(alist)>1:

        mid = len(alist)//2

        lefthalf = alist[:mid]

        righthalf = alist[mid:]



        mergeSort(lefthalf)

        mergeSort(righthalf)



        i=0

        j=0

        k=0

        while i < len(lefthalf) and j < len(righthalf):

            if lefthalf[i] < righthalf[j]:

                alist[k]=lefthalf[i]

                i=i+1

            else:

                alist[k]=righthalf[j]

                j=j+1

            k=k+1



        while i < len(lefthalf):

            alist[k]=lefthalf[i]

            i=i+1

            k=k+1



        while j < len(righthalf):

            alist[k]=righthalf[j]

            j=j+1

            k=k+1

        if (len(alist) % 5000==0):

            elePlot.extend([len(alist)])

            timePlot.extend([(time() - t1)])

    #print("Merging ",alist)



#alist = [54,26,93,17,77,31,44,55,20]

alist = BitArray(60000)

mergeSort(alist)

#print(alist)





#avg1 = (time() - t1) / 60000 * 1000

#print("Average runtime per 1000th sorted element is ", "{0:.2f}".format(avg1), "seconds")

print("The graph with merge-sort for 60k elements of bits")

plt.xlabel('Running Time /s') 

plt.ylabel('Input Size') 

plt.plot(timePlot, elePlot, label =' ') 

plt.grid() 

plt.legend() 

plt.show() 

#Quick-Sort



timePlot = []

elePlot = []

t1 = time()



def partition(arr,low,high): 

    i = ( low-1 )         # index of smaller element 

    pivot = arr[high]     # pivot 

  

    for j in range(low , high): 

  

        # If current element is smaller than or 

        # equal to pivot 

        if   arr[j] <= pivot: 

          

            # increment index of smaller element 

            i = i+1 

            arr[i],arr[j] = arr[j],arr[i] 

  

    arr[i+1],arr[high] = arr[high],arr[i+1] 

    return ( i+1 ) 

  

# The main function that implements QuickSort 

# arr[] --> Array to be sorted, 

# low  --> Starting index, 

# high  --> Ending index 

  

# Function to do Quick sort 

def quickSort(arr,low,high): 

    if low < high: 

        count =+ 1

        # pi is partitioning index, arr[p] is now 

        # at right place 

        pi = partition(arr,low,high) 

  

        # Separately sort elements before 

        # partition and after partition 

        quickSort(arr, low, pi-1) 

        quickSort(arr, pi+1, high) 

    

# Driver code to test above 

arr = [10, 7, 8, 9, 1, 5] 

#arr = BitArray(60)

n = len(arr) 

quickSort(arr,0,n-1) 

print ("Sorted array is:") 

for i in range(n): 

    print ("%d" %arr[i]), 





#elePlot.extend([1])

#elePlot.extend([len(arr)])

#timePlot.extend([(time() - t1)])



#print("The graph with quick-sort for 60k elements of bits")

#plt.xlabel('Running Time /s') 

#plt.ylabel('Input Size') 

#plt.plot(timePlot, elePlot, label =' ') 

#plt.grid() 

#plt.legend() 

#plt.show() 
def union(A, B):

    result = []

    a = 0

    b = 0

    while a < len(A) and b < len(B):

        if result:

            if a < len(A) and result[-1] == A[a]:

                a += 1

                continue

            if b < len(B) and result[-1] == B[b]:

                b += 1

                continue

        if a >= len(A):

        #A is out of items, just insert B.

            result.append(B[b])

        elif b >= len(B):

            result.append(A[a])

        elif A[a] < B[b]:

        #A[a] is lower.

            result.append(A[a])

        else:

            result.append(B[b])

        #return result

        print(result)

A = [0, 1, 3, 3, 3, 5, 7]

B = [2, 4, 5, 7, 7]

union(A,B)
#Use quick sort again



def partition(arr,low,high): 

    i = ( low-1 )         # index of smaller element 

    pivot = arr[high]     # pivot 

  

    for j in range(low , high): 

  

        # If current element is smaller than or 

        # equal to pivot 

        if   arr[j] <= pivot: 

          

            # increment index of smaller element 

            i = i+1 

            arr[i],arr[j] = arr[j],arr[i] 

  

    arr[i+1],arr[high] = arr[high],arr[i+1] 

    return ( i+1 ) 

  

# The main function that implements QuickSort 

# arr[] --> Array to be sorted, 

# low  --> Starting index, 

# high  --> Ending index 

  

# Function to do Quick sort 

def quickSort(arr,low,high): 

    if low < high: 

        count =+ 1

        # pi is partitioning index, arr[p] is now 

        # at right place 

        pi = partition(arr,low,high) 

  

        # Separately sort elements before 

        # partition and after partition 

        quickSort(arr, low, pi-1) 

        quickSort(arr, pi+1, high) 

    

# Driver code to test above 

arr = [0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,0] 

#arr = BitArray(60)

n = len(arr) 

quickSort(arr,0,n-1) 

print ("Sorted array is:") 

for i in range(n): 

    print ("%d" %arr[i]), 
votes = [1,2,3,4,5,5,3,4,4,3,2,2,1,2,2,2,3,4,5,4,3,3,2]

candidates = ["Emmanuel" , "Donald", "Theresa", "Angela", "Vladimir"]



def counter():

    candnr = 0

    leader = 0

    curr = 0

    for i in candidates:

        candnr = candnr + 1

        current = votes.count(candnr)

        if current > leader:

            newpresident = i

            leader = current

    print("The new president is going to be:" ,newpresident)

        

counter()

print("1:",votes.count(1))

print("2:",votes.count(2))

print("3:",votes.count(3))

print("4:",votes.count(4))

print("5:",votes.count(5))
def bucket_sort(alist):

    largest = max(alist)

    length = len(alist)

    size = largest/length

 

    buckets = [[] for _ in range(length)]

    for i in range(length):

        j = int(alist[i]/size)

        if j != length:

            buckets[j].append(alist[i])

        else:

            buckets[length - 1].append(alist[i])

 

    for i in range(length):

        insertion_sort(buckets[i])

 

    result = []

    for i in range(length):

        result = result + buckets[i]

 

    return result

 

def insertion_sort(alist):

    for i in range(1, len(alist)):

        temp = alist[i]

        j = i - 1

        while (j >= 0 and temp < alist[j]):

            alist[j + 1] = alist[j]

            j = j - 1

        alist[j + 1] = temp

 

 

#alist = input('Enter the list of (nonnegative) numbers: ').split()

alist = [1,2,3,4,5,5,3,4,4,3,2,2,1,2,2,2,3,4,5,4,3,3,2]

alist = [int(x) for x in alist]

sorted_list = bucket_sort(alist)

print('Sorted list: ', end='')

print(sorted_list)


# Python program for implementation of MergeSort 

def mergeSort(arr): 

    if len(arr) >1: 

        mid = len(arr)//2 #Finding the mid of the array 

        L = arr[:mid] # Dividing the array elements  

        R = arr[mid:] # into 2 halves 

  

        mergeSort(L) # Sorting the first half 

        mergeSort(R) # Sorting the second half 

  

        i = j = k = 0

          

        # Copy data to temp arrays L[] and R[] 

        while i < len(L) and j < len(R): 

            if L[i] < R[j]: 

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

  

# Code to print the list 

def printList(arr): 

    for i in range(len(arr)):         

        print(arr[i],end=" ") 

    print() 

  

# driver code to test the above code 

if __name__ == '__main__': 

    arr = [1000, 80, 10, 50, 70, 60, 90, 20]  

    print ("Given array is", end="\n")  

    printList(arr) 

    mergeSort(arr) 

    print("Sorted array is: ", end="\n") 

    printList(arr) 

  
