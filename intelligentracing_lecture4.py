## This is course material for Introduction to Python Scientific Programming

## Class 2 Example code: reverse_list_time.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



import time



# We have to create a very long list and very large repeat time to be able to time

List = list('abcdefghijklmnopqrstuvwxyz' * 10)

repeat_time = 1000000



# Test total time running while repeat_time 

begin_time = time.time()

for i in range(repeat_time):

    reversed_list = []

    index = len(List) # index point to last character + 1

    while index>0:

        index = index -1

        reversed_list.append(List[index])

elapsed_time = time.time() - begin_time

print('WHILE METHOD: Elapsed time: %.2f seconds' % elapsed_time)



# Test total time running slicing repeat_time 

begin_time = time.time()

for i in range(repeat_time):

    reversed_list = []

    reversed_list = List[::-1]

elapsed_time = time.time() - begin_time

print('SLICE METHOD: Elapsed time: %.2f seconds' % elapsed_time)



# Test total time running in-place reverse() repeat_time 

begin_time = time.time()

for i in range(repeat_time):

    reversed_list=List.copy()

    reversed_list.reverse()

elapsed_time = time.time() - begin_time

print('REVERSE METHOD: Elapsed time: %.2f seconds' % elapsed_time)



# Test total time running in-place reverse() without copy repeat_time 

begin_time = time.time()

for i in range(repeat_time):

    List.reverse()

elapsed_time = time.time() - begin_time

print('REVERSE METHOD: Elapsed time: %.2f seconds' % elapsed_time)
## This is course material for Introduction to Python Scientific Programming

## Class 2 Example code: reverse_list.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



List = list("Python")



# Method 1, using loop

reversed_list = []

index = len(List) # index point to last character + 1

while index>0:

    index = index -1

    reversed_list.append(List[index])



print(reversed_list)



# Method 2: extended slicing

reversed_list = List[::-1]

print(reversed_list)



# Method 3: use object method

reversed_list = List.copy()

reversed_list.reverse()  # reverse in-place

print(reversed_list)
## This is course material for Introduction to Python Scientific Programming

## Class 4 Example code: reverse_string.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use





input_string = "Python"



# Method 1, using loop

reversed_string = ""

index = len(input_string) # index point to last character + 1

while index>0:

    index = index -1

    reversed_string = reversed_string + input_string[index]



print(reversed_string)



# Method 2, using slicing

reversed_string = input_string[len(input_string)-1::-1]

print(reversed_string)



sliced_string = input_string[::-1]

print(sliced_string)