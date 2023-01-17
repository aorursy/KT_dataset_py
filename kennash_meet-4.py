# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
type(3)

print("hello")



#functions take arguments - It's whats in the parentheis

#What they do things with



help(print) #Help is used when you want to figure out what something can do

print("Hello", "Hi", sep = "3")





groceries = ["1. milk", "2. Bread",  "3. Eggs"]

print(*groceries)



newlist = ["h", "e", 'l', "l", "o"]

print(*newlist, sep = "") #don't worry about the asterisk for now, it just unpacks the list (we can talk about that later)



#help(int)  # Running this would give you a lot of information about the integer type



def newfunction(a):

    return a+10 #"Return" saves a certain value to the function



print(newfunction(30))



def function_two():

    return "This did not need an argument"



print(function_two())









def min_diff_func(a, b, c):

    diff_list = [abs(a-b), abs(a-c), abs(b-c)]

    return min(diff_list)



z = min_diff_func(10, 12, 200) 



def min_diff_func(a = 10, b = 12, c = 100):

    diff_list = [abs(a-b), abs(a-c), abs(b-c)]

    return min(diff_list)



min_diff_func(1, 2, 3)





print(1 in [2, 3])







def can_run_for_president(age, is_natural_born_citizen):

    

    # The US Constitution says you must be a natural born citizen *and* at least 35 years old

    return (age >= 35) and is_natural_born_citizen



print(can_run_for_president(19, True))

print(can_run_for_president(55, False))

print(can_run_for_president(55, True))





print(True or (True and False))



umbrella = True

have_hood = False

less_than_5 = False

print(umbrella or have_hood and less_than_5)



print((not False) == (False))



def sign_id(num):

    if num > 0:

        print("num is positive")

    elif num < 0:

        print("num is negative")

    else:

        print("num is zero")



sign_id(6)





def check(username, admin_access):

    if username == "Ken" and admin_access == True:

        return("You are logged in and an admin")

    elif username == "Ken":

        return("You are logged in")

    elif admin_access == True:

        return("You are a teacher")

    else:

        return("You cannot log in")

        

check("Ken", True)

    
