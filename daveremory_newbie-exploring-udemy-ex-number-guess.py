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
#BELOW IS A COPY OF THE WRONG FIRST CODE FROM UDEMY EXAMPLE - CPOIED IN FROM PYTHON IDLE

#imports a set of libraries for generating random numbers

import random



#def is a keyword for calling a method, our method, which is simply a block fo code we've written

def ask_user_and_check_number():

    user_number = int(input("enter number between 0-9: "))

    if user_number in magic_numbers:

        print ("right on Dave, you're kicking ass now")

    if user_number not in magic_numbers:

        print ("wrong number, try again")



magic_numbers = [random.randint(0,9), random.randint(0,9)]



def run_program_x_times(chances):

    for attempt in range(chances):

        print ("This is attempt {}".format(attempt))

# below we are going to call our method   

        ask_user_and_check_number()

    

run_proram_x_times(1)
#HERE IS SECOND ATTEMPT AT CODE, CORRECT OUTPUT

#imports a set of libraries for generating random numbers

import random



#def is a keyword for calling a method, our method, which is simply a block fo code we've written

def ask_user_and_check_number():

    user_number = int(input("enter number between 0-9: "))

    if user_number in magic_numbers:

        print ("right on Dave, you're kicking ass now")

    if user_number not in magic_numbers:

        print ("wrong number, try again")



magic_numbers = [random.randint(0,9), random.randint(0,9)]

chances = 3

for attempt in range(chances):

    print ("This is attempt {}".format(attempt))

# below we are going to call our method    

    ask_user_and_check_number()
# final code with help

import random

def ask_user_and_check_number():

    user_number = int(input("enter number between 0-9: "))

    if user_number in magic_numbers:

        print ("right on Dave, you're kicking ass now")

    if user_number not in magic_numbers:

        print ("wrong number, try again")

magic_numbers = [random.randint(0,9), random.randint(0,9)]

def run_program_x_times(user_attempts):

    for attempt in range(user_attempts):

        # first attempt to run with new method, print command was not indented; indented now

        print ("This is attempt {}".format(attempt))

        ask_user_and_check_number()

    

user_attempts = int(input("Enter number of attempts: "))

run_program_x_times(user_attempts)



    
#attempot without help

import random

#this line was hard to remember random.randint, not: int(rand(x,y))

magic_number = [random.randint(0,9),random.randint(0,9)]

def ask_user_to_check_number():

    user_number = int(input("Please Enter a Number:"))

# no need to indent again here, undder the variable

    if user_number in magic_number:

        print("You got the code and the number right, Dave")

    if user_number not in magic_number:

        print("You got the code right, Dave, but not the number. Try again.")

def times_user_wants_to_try(user_attempts):

    user_attempts = int(input("Enter Number of Trials: "))

    for attempts in range (user_attempts):

    # here you didnt call out the variable attempts in paren; also its {}".format..... not {}."format <<-- notice period formatting

        print("Number of attempts {}".format(attempts))

        # below, the first method ask user to check number is contained int the second method times user wants to try... which wasnt scripted to be called yet

        ask_user_to_check_number

#TypeError: times_user_wants_to_try() missing 1 required positional argument: 'user_attempts'<<-- input variable "user_attempts"

times_user_wants_to_try(user_attempts)

#And... on to the next panel
#attempt 2 without help

import random

magic_number = [random.randint(0,9),random.randint(0,9)]

user_number = int(input("Please Enter a Number: "))

# this line cannot be here, it must be defined in the "times user wants to try method" since it contains an input

user_attempts = int(input("Enter Number of Trials: "))



def ask_user_to_check_number():

    if user_number in magic_number:

        print("You got the code and the number right, Dave")

    if user_number not in magic_number:

        print("You got the code right, Dave, but not the number. Try again.")



def times_user_wants_to_try(user_attempts):

    for attempts in range (user_attempts):

        print("Number of attempts {}".format(attempts))

        ask_user_to_check_number

times_user_wants_to_try(user_attempts)

#attempt 3 without help - aattempt to add an end of game flag if number of attempts = number of trials

import random

magic_number = [random.randint(0,9),random.randint(0,9)]

user_attempts = int(input("Enter Number of Trials: "))



def ask_user_to_check_number():

    user_number = int(input("Please Enter a Number: "))

    if user_number in magic_number:

        print("You got the code and the number right, Dave!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if user_number not in magic_number:

        print("You got the code right, Dave, but not the number. Try again.")



def times_user_wants_to_try(user_attempts):

    for attempts in range (user_attempts):

        print("Number of attempts {}".format(attempts))

        ask_user_to_check_number()

times_user_wants_to_try(user_attempts)
