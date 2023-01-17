# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import timeit
import numpy as np
import random
random.seed(42)

from math import factorial
# Any results you write to the current directory are saved as output.
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def timer(func, *args):
    wrapped = wrapper(func, *args)
    time = timeit.timeit(wrapped,number=10000)
    return("Time of execution is {} ms".format(time))
# Example from Ravi Ojha's post (see link above). 
# Adds 2 numbers
def O1_add(n1, n2):
    return (n1 + n2)
# No matter what the input, the function executes in one step, so roughly the same time complexity
for n in range(1,6):
    print(n,",",n + random.randint(1,int(1e10)))
    print(timer(O1_add, int(n), int(n) + random.randint(1,int(1e10))))
    print()
# Checks whether a number is even or odd by checking last digit of binary representation
def O1_odd_check(num):
    is_odd = False
    if num & 1 == 1:
        is_odd = True
    return is_odd
check_lst = [1,5,8,82,101]
for num in check_lst:
    print(num,"::",O1_odd_check(num),"::",timer(O1_odd_check, num))
# Finds an item in an unsorted list
def On_simple_search(lst,number):
    is_found = False
    for num in lst:
        if num == number:
            is_found = True
    return is_found
lst1 = range(5)
lst2 = range(500)
lst3 = range(50000)

num1 = 2
num2 = -50
num3 = 4000
print(On_simple_search(lst1,num1),"::",timer(On_simple_search,lst1,num1))
print(On_simple_search(lst2,num2),"::",timer(On_simple_search,lst2,num2))
print(On_simple_search(lst3,num3),"::",timer(On_simple_search,lst3,num3))
def Ologn_binary_search(list,number):
    first = 0
    last = len(list) - 1
    is_found = False
    while first <= last and not is_found:
        mid = (first + last)//2
        if list[mid] == number:
            is_found = True
        else:
            if number < mid:
                last = mid - 1
            else:
                first = mid + 1
    return is_found
lst1 = range(5)
lst2 = range(500)
lst3 = range(50000)

num1 = 2
num2 = -50
num3 = 4000
print(Ologn_binary_search(lst1,num1),"::",timer(Ologn_binary_search,lst1,num1),"::","log value = {}".format(np.log2(len(lst1))))
print(Ologn_binary_search(lst2,num2),"::",timer(Ologn_binary_search,lst2,num2),"::","log value = {}".format(np.log2(len(lst2))))
print(Ologn_binary_search(lst3,num3),"::",timer(Ologn_binary_search,lst3,num3),"::","log value = {}".format(np.log2(len(lst3))))
def Onlogn_merge_sort(sequence):
    if len(sequence) < 2:
        return sequence
    
    m = len(sequence) // 2
    return Onlogn_merge(Onlogn_merge_sort(sequence[:m]), Onlogn_merge_sort(sequence[m:]))


def Onlogn_merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]

    return result
array = [4, 2, 3, 8, 8, 43, 6,1, 0]
ar = Onlogn_merge_sort(array)
print (" ".join(str(x) for x in ar))
lst1 = [4,2,3,8,8,43,6,1,0,83]
lst2 = []
for i in range(100):
    lst2.append(random.randint(0,i))
print("Sorted lst1:: ",Onlogn_merge_sort(lst1))
print(timer(Onlogn_merge_sort,lst1)," :: nlogn ~= {}".format(len(lst1)*np.log2(len(lst1))))

print("Sorted lst2:: ",Onlogn_merge_sort(lst2))
print(timer(Onlogn_merge_sort,lst2)," :: nlogn ~= {}".format(len(lst2)*np.log2(len(lst2))))
def On2_bubble_sort(lst):
    for i in range(len(lst)-1):
        for j in range(len(lst)-1-i):
            if lst[j] > lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst
lst1 = [4,2,3,8,8,43,6,1,0,83]
lst2 = []
for i in range(100):
    lst2.append(random.randint(0,i))
print("Sorted lst1:: ",On2_bubble_sort(lst1))
print(timer(On2_bubble_sort,lst1)," :: n^2 ~= {}".format(len(lst1)**2))

print("Sorted lst2:: ",On2_bubble_sort(lst2))
print(timer(On2_bubble_sort,lst2)," :: n^2 ~= {}".format(len(lst2)**2))
# Sum of a Fibonacci series up to the nth term
def o2n_fibonacci(n):
    if n<2:
        return n
    return o2n_fibonacci(n-1) + o2n_fibonacci(n-2)
for n in range(2,12,2):
    print("Series sum for {} is {}".format(n,o2n_fibonacci(n))," :: ",timer(o2n_fibonacci,n)," :: 2^n = {}".format(2**n))
def onfac_perm(a, k=0):
    if k==len(a):
#         print(a) # Commendted out for display purposes
        pass
    else:
        for i in range(k, len(a)):
            a[k],a[i] = a[i],a[k]
            onfac_perm(a, k+1)
            a[k],a[i] = a[i],a[k]
lst1 = [1,2,]
lst2 = [1,2,3,4]
lst3 = [1,2,3,4,5,6]

print("List of {} items :: ".format(len(lst1)), timer(onfac_perm,lst1), " :: factorial {} is {}".format(len(lst1),factorial(len(lst1))))
print("List of {} items :: ".format(len(lst2)), timer(onfac_perm,lst2), " :: factorial {} is {}".format(len(lst2),factorial(len(lst2))))
print("List of {} items :: ".format(len(lst3)), timer(onfac_perm,lst3), " :: factorial {} is {}".format(len(lst3),factorial(len(lst3))))
