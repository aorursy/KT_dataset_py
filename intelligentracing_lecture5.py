## This is course material for Introduction to Python Scientific Programming

## Class 5 Example code: 10_factorial.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



index = 0

fac = 1

while (index<=10):

    index = index + 1

    fac = fac * index

else:

    print(fac)
## This is course material for Introduction to Python Scientific Programming

## Class 5 Example code: find_prime.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import math

import sys



print("Please provide an integer limit for finding prime numbers: ")

int_limit = input()



try:  # Try to catch posible error when int_limit is not an integer

    int_limit = int(int_limit)

except: # When runtime error occurs, except will be executed

    print("Not a valid integer input. Exit!")

else: # else means no runtime error

    if int_limit<2:   # Testing prime only meaningful for numbers >=2

        print("No prime number within the range")

        sys.exit()

    for n in range(2, int_limit+1):

        if n==2:  # Number 2 is defined as a the first prime number 

            print(n, 'is a prime number')

            continue



        # if n is > 2, test if n can be divided by two nontrivial integers

        for x in range(2, math.ceil(math.sqrt(n)+1)):

            if n % x == 0:

                print(n, 'equals', x, '*', n//x)

                break

        else:

            # loop fell through without finding a factor

            print(n, 'is a prime number')
import math

import sys



print("Please provide an integer limit for finding prime numbers: ")

int_limit = input()

is_finished = True

try:  # Try to catch posible error when int_limit is not an integer

    int_limit = int(int_limit)

except: # When runtime error occurs, except will be executed

    print("Not a valid integer input. Exit!")

else: # else means no runtime error

    if int_limit<2:   # Testing prime only meaningful for numbers >=2

        print("No prime number within the range")

        sys.exit()

    for n in range(2, int_limit+1):

        if n==2:  # Number 2 is defined as a the first prime number 

            print(n, 'is a prime number')

            continue



        # if n is > 2, test if n can be divided by two nontrivial integers

        for x in range(2, math.ceil(math.sqrt(n)+1)):

            if n % x == 0:

                is_finished = False

                break

        if is_finished == True:

            # loop fell through without finding a factor

            print(n, 'is a prime number')

        else:

            is_finished = True

            print(n, 'equals', x, '*', n//x) 
## This is course material for Introduction to Python Scientific Programming

## Class 5 Example code: for_continue.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



animal_list = ['dog', 'cat','fish','pony','parrot','leopard','frog','mouse','snake']



index = 0

adopted_animals=[]

for i in animal_list:

    print('Would you like to adopt a '+ i + '? [Y/N]')

    answer = input('Would you like to adopt a '+ i + '? [Y/N]')

    if answer.lower()!='y':

        continue

    

    adopted_animals.append(i)



if not adopted_animals:

    print('Your adoption list is empty. See you next time!')

else:

    print('We will have your ' + str(adopted_animals) + ' ready for pick up!')
## This is course material for Introduction to Python Scientific Programming

## Class 5 Example code: while_not_quit.py

## Author: Allen Y. Yang,  Intelligent Racing Inc.

##

## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use



animal_list = ['dog', 'cat','fish','pony','parrot','leopard','frog','mouse','snake']



index = 0

while True:

    print('Would you like to adopt a '+ animal_list[index%9]+ '? [Y/N]')

    answer = input('Would you like to adopt a '+ animal_list[index%9]+ '? [Y/N]')

    if answer.lower()=='y':

        break

    index += 1



print('We will have your ' + animal_list[index%9] + ' ready for pick up!')