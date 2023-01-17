import random

import timeit

import time



#**************************************************#

            ## ~ELEMANLARI RASTGELE~ ##

    

start=time.time()



list1=[]

for r1 in range(100):



    r1=random.randint(1,100)

    list1.append(r1)



print("\t\t\t ELEMANLARI RASTGELE \n")

print(list1,"\n")



stop=time.time()

print("Work Time: ",stop-start,"\n")



#**************************************************#

            ## ~ELEMANLARI SIRALI~ ##



start=time.time()



list2=[]

for r2 in range(100):



    r2=random.randint(1,100)

    list2.append(r2)

    

list2.sort()

print("\t\t\t ELEMANLARI SIRALI \n")

print(list2,"\n")



stop=time.time()

print("Work Time: ",stop-start,"\n")



#**************************************************#

            ## ~ELEMANLARI TERS SIRALI~ ##



start=time.time()



list3=[]

for r3 in range(100):

    

    r3=random.randint(1,100)

    list3.append(r3)



list3.sort(reverse=True)

print("\t\t\t ELEMANLARI TERS SIRALI \n")

print(list3,"\n")



stop=time.time()

print("Work Time: ",stop-start,"\n")







keep1=list1

keep2=list2

keep3=list3

long=100;

#**************************************************#

            ## ~İNSERTİON SORT~ ##

start=time.time()



def insertion_sort(List):

    for i in range(1,long):

        key=List[i]

        j=i-1

        while j >=0 and key < List[j]:

            List[j+1] = List[j]

            j-=1

        List[j+1] = key

        

stop=time.time()

print("Creation Time: ",stop-start,"\t İnsertion Sort","\n")        

#**************************************************#

            ## ~SELECTİON SORT~ ##

start=time.time()



def selection_sort(List):

    for i in range(long): 

        min_index = i 

        for j in range(i+1,long): 

            if List[min_index] > List[j]: 

                min_index = j 

              

    # Swap the found minimum element with  

    # the first element         

        List[i], List[min_index] = List[min_index], List[i]

        

stop=time.time()

print("Creation Time: ",stop-start,"\t Selection Sort","\n")        

#**************************************************#

            ## ~BUBBLE SORT~ ##

start=time.time()



def bubble_Sort(List):

    for i in range(long):

        for j in range(0, long-i-1):

            if List[j] > List[j+1] :

                List[j], List[j+1] = List[j+1], List[j]

                

stop=time.time()

print("Creation Time: ",stop-start,"\t Bubble Sort","\n")        

#**************************************************#

            ## ~HEAP SORT~ ##

start=time.time()



def heapify(List,long, i): 

    largest = i  # Initialize largest as root 

    l = 2 * i + 1     # left = 2*i + 1 

    r = 2 * i + 2     # right = 2*i + 2 

  

    # See if left child of root exists and is 

    # greater than root 

    if l < long and List[i] < List[l]: 

        largest = l 

  

    # See if right child of root exists and is 

    # greater than root 

    if r < long and List[largest] < List[r]: 

        largest = r 

  

    # Change root, if needed 

    if largest != i: 

        List[i],List[largest] = List[largest],List[i]  # swap 

  

        # Heapify the root. 

        heapify(List, long, largest) 



def heap_Sort(List): 

    for i in range(long, -1, -1): 

        heapify(List, long, i) 

    for i in range(long-1, 0, -1): 

        List[i], List[0] = List[0], List[i]   

        heapify(List, i, 0)

        

stop=time.time()

print("Creation Time: ",stop-start,"\t Heap Sort","\n")        

print("\t\t  İnsertion Sort")

start=time.time()

insertion_sort(list1)

stop=time.time()

print("List 1\n   Sort Time: ",stop-start,"\n")



start=time.time()

insertion_sort(list2)

stop=time.time()

print("List 2\n   Sort Time: ",stop-start,"\n")



start=time.time()

insertion_sort(list3)

stop=time.time()

print("List 3\n   Sort Time: ",stop-start,"\n")
print("\t\t  Selection Sort")

start=time.time()

selection_sort(list1)

stop=time.time()

print("List 1\n   Sort Time: ",stop-start,"\n")



start=time.time()

selection_sort(list2)

stop=time.time()

print("List 2\n   Sort Time: ",stop-start,"\n")



start=time.time()

selection_sort(list3)

stop=time.time()

print("List 3\n   Sort Time: ",stop-start,"\n")
print("\t\t  Bubble Sort")

start=time.time()

bubble_Sort(list1)

stop=time.time()

print("List 1\n   Sort Time: ",stop-start,"\n")



start=time.time()

bubble_Sort(list2)

stop=time.time()

print("List 2\n   Sort Time: ",stop-start,"\n")



start=time.time()

bubble_Sort(list3)

stop=time.time()

print("List 3\n   Sort Time: ",stop-start,"\n")
print("\t\t  Heap Sort")

start=time.time()

heap_Sort(list1)

stop=time.time()

print("List 1\n   Sort Time: ",stop-start,"\n")



start=time.time()

heap_Sort(list2)

stop=time.time()

print("List 2\n   Sort Time: ",stop-start,"\n")



start=time.time()

heap_Sort(list3)

stop=time.time()

print("List 3\n   Sort Time: ",stop-start,"\n")