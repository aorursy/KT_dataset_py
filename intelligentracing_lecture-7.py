## This is course material for Introduction to Python Scientific Programming

## Class 7 Example code: compare_sort.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



import random

import time



def insert_sort(input_list):

    ''' A custom function to sort number sequences using insert sort

    Parameters:

    Input:  input_list  - Expecting a list of numerical numbers



    Output: input_list  - sorted list

    '''

    if type(input_list)!=list:

        input_list = list(input_list)



    for index in range(1, len(input_list)):

    

        # Compare and sort elements one by one

        current = input_list[index]



        # Verify the type of each element

        if type(current)!=int and type(current)!=float:

            current = float(current)



        # Insert to previous sorted sub-list

        while  (index>0 and input_list[index-1]>current):

            # Insert iteratively until insert condition is False

            input_list[index] = input_list[index-1]

            input_list[index-1] = current

            index -=1

    

    return input_list



# Generate a sufficiently long list for sorting

sample_count = 10000

random_input = random.sample(range(0, sample_count),sample_count)



# ******** Method 1: Insert Sort ********

print('*** Insert Sort ***')

result = random_input.copy()

begin_time = time.time()

insert_sort(result)



# tic-toc 

elapsed_time = time.time() - begin_time

print(elapsed_time)

print(result[0:20])



# ******** Method 2: Built-in Timsort ******

print('*** Python Sort ***')

result = random_input.copy()

begin_time = time.time()

result.sort()



# tic-toc 

elapsed_time = time.time() - begin_time

print(elapsed_time)

print(result[0:20])
## This is course material for Introduction to Python Scientific Programming

## Class 7 Example code: fibonacci.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



def fibonacci(n):

    ''' A recursive function to calculate the Fibonacci number

    Parameters:

    - Input: n an integer >= 0

    - Output: Integer Fibonacci number

    '''



    if type(n)!= int:

        raise TypeError('Incorrect Fibonacci argument type.')

    elif n<0:

        raise ValueError('Fibonacci argument must be greater than zero.')



    if n == 0:

        return 0

    elif n==1:

        return 1

    else:

        return fibonacci(n-1) + fibonacci(n-2)





print(fibonacci(19))



print(fibonacci(-2))
## This is course material for Introduction to Python Scientific Programming

## Class 7 Example code: func_test.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



def func_test(L = ['a', 'b'], S = 'ab'):

    ''' Append and return one mutable list and one immutable string. '''

    L.append('c')

    S = S + 'c'

    return L, S



def func_test1(L , S ):

    ''' Append and return one mutable list and one immutable string. '''

    L.append('c')

    S = S + 'c'

    return S



def func_test2(L):

    ''' Append and return one mutable list and one immutable string. '''

    global S



    L= L.append('c')

    S = S + 'c'



L = ['a', 'b']; S = 'ab'

S = func_test1(L, S)

print('{0}, {1}'.format(L,S))



L = ['a', 'b']; S = 'ab'

func_test2(L)

print('{0}, {1}'.format(L,S))
## This is course material for Introduction to Python Scientific Programming

## Class 7 Example code: functions_say_hello.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



def print_hello_world():

    ''' The function prints Hello World! string. '''



    print('Hello World!') 



def print_string(argin):

    """

    The function converts argin to string and print. 

    Input:

        argin: argin will be converted to string

    Output: None

    """



    try:

        string = str(argin)

    except:

        print('Illegal input! exit')

    else: 

        print(string) 



def hello_world():

    """The function returns Hellow World! string. """



    return('Hello World!')
## This is course material for Introduction to Python Scientific Programming

## Class 7 Example code: merge_sort.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



import random

import time



def insert_sort(input_list, reverse = False):

    ''' A custom function to sort number sequences using insert sort

    Parameters:

    Input:  input_list  - Expecting a list of numerical numbers

            order       - Ascending or descending order, default = 0



    Output: status      - Boolean: True or False

            input_list  - sorted list if status is True

    '''

    if type(reverse)!=bool:

        return False



    for index in range(len(input_list)):

    

        # Compare and sort elements one by one

        current = input_list[index]



        # Verify the type of each element

        if type(current)!=int and type(current)!=float:

            return False



        # Insert to previous sorted sub-list

        # Insert condition based on order

        if reverse == 0:

            while_condition = (index>0 and input_list[index-1]>current)

        else:

            while_condition = (index>0 and input_list[index-1]<current)

        while while_condition:

            # Insert iteratively until insert condition is False

            input_list[index] = input_list[index-1]

            input_list[index-1] = current

            index -=1

            if reverse == 0:

                while_condition = (index>0 and input_list[index-1]>current)

            else:

                while_condition = (index>0 and input_list[index-1]<current)

    

    return True



def merge_sort(input_list):

    ''' Merge sort function using recursion

    Parameters:

    Input:  input_list  - a list of numerical numbers



    Output: input_list  - sorted list

    '''

    

    # Deploy Divide-and-Conquer

    if len(input_list)>1:

        mid = len(input_list)//2

        left_half = input_list[:mid]

        right_half = input_list[mid:]



        # Recursively sort left and right sub-lists

        merge_sort(left_half)

        merge_sort(right_half)



        # Merging left_half and right_half 

        left_pointer=0

        right_pointer=0

        merged_pointer=0

        while left_pointer < len(left_half) and right_pointer < len(right_half):

            if left_half[left_pointer] <= right_half[right_pointer]:

                input_list[merged_pointer]=left_half[left_pointer]

                left_pointer=left_pointer+1

            else:

                input_list[merged_pointer]=right_half[right_pointer]

                right_pointer=right_pointer+1

            merged_pointer=merged_pointer+1



        while left_pointer < len(left_half):

            input_list[merged_pointer]=left_half[left_pointer]

            left_pointer=left_pointer+1

            merged_pointer=merged_pointer+1



        while right_pointer < len(right_half):

            input_list[merged_pointer]=right_half[right_pointer]

            right_pointer=right_pointer+1

            merged_pointer=merged_pointer+1



# Generate a sufficiently long list for sorting

sample_count = 10000

random_input = random.sample(range(0, sample_count),sample_count)



# ******** Method: Merge Sort ********

print('*** Merge Sort ***')

result = random_input.copy()

begin_time = time.time()

merge_sort(result)



# tic-toc 

elapsed_time = time.time() - begin_time

print(elapsed_time)

print(result[0:20])