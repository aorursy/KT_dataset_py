# List of magic functions

%lsmagic
%%writefile quiz.py

from random import randint



#how big a number should we guess? 

max_number = 12

first_line = "Guess a number between 1 and %d" % max_number

print(first_line)



number = randint(1, max_number)



not_solved = True



#keep looping unil we guess correctly

while not_solved:

    answer = int(input('?'))

    you_said = "You typed %d" % answer

    print (you_said)

    if answer > number:

        print ("The number is lower")

    elif answer < number:

       print ("The number is higher")

    else:

        print ("You got it right")

        not_solved = False
# Runing external code

%run quiz.py

# %load imports.py

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%who
%who_ls
%reset 

# or %reset -f if no prompt desired

# Defining the alias named parts where there are two arguments as first and second

%alias parts echo First %s Second %s

#You can use the alias here as below

%parts "Call this number 1234." "Leave message if nobody there."
# Unset the alias parts

%unalias parts
%timeit sum(range(1000))