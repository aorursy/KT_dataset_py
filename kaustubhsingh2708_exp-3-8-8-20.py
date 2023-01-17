import random
import numpy as np
a=np.array([])
for i in range(15):
    n=random.randint(0,100)
    a=np.append(a,[n])
print(a)
def sum_of_array():
    sum1=0
    for i in range(15):
        sum1+=a[i]
    print("Sum of array elements is:",sum1)
def product_of_array():
    prod=1
    for i in range(15):
        prod*=a[i]
    print("product of elements is:",prod)
sum_of_array()
product_of_array()
    
import random
import numpy as np
import time
a=np.array([])
for i in range(15):
    n=random.randint(0,100)
    a=np.append(a,[n])
print(a)
def add_list():
    list1=[]
    for i in range(15):
        list1.append(a[i])
    return list1
def prod_of_array():
    start_time=time.time()
    prod=1
    for i in range(15):
        prod*=a[i]
    print("product of elements in array is:",prod)
    
    time_array=time.time()-start_time
    return time_array
def prod_of_list(list1):
    start_time=time.time()

    prod=1
    for i in range(15):
        prod*=list1[i]
    print("product of elements in list is:",prod)

    time_list=time.time()-start_time
    return time_list
def time_comp(a,b):
    print("time taken by array:",a)
    
    print("time taken by list:",b)
    #calculation
    z=((b-a)/b)*100
    print("time taken by array is faster than list by",z,"percentage")
    
    
add_list()
b=prod_of_array()
print(b)
list1=add_list()
prod_of_list(list1)
a=prod_of_list(list1)

time_comp(a,b)