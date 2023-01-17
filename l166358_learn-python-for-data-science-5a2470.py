# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
#Let's start with data types
# Textual data is stored in string data type
textual_data = 'Hello World'
#Let's print the type of textual_data
print(type(textual_data))
#Numbers are stored in integer data type
number_data = 5
print(type(number_data))
#Numbers with decimal point are stored in float type
float_data = 3.3
print(type(float_data))
#Hmm, can we have an array of these data types
# yes, You can store a list of these data types in list
list_type = [1, 'text', 2, 'another text']
print(type(list_type))
# We have tuples too and yes they are immutable, lists are mutable
tuple_type = (1, 'text', 2, 'another text')
print(type(tuple_type))
# Lets print our list
print(list_type)
# Let's append something to list and print again
list_type.append('I am feeling lucky to use python')
print(list_type)
# I know what you are thinkink now
# Yes, you can iterate a list very easily
for item in list_type:
    print(item)
# Let's slice and dice our list now, shall we?
print(list_type[-1])
print(list_type[0])
print(list_type[-3:-1])
print(list_type[:2])
print(list_type[2:])
# We have a very powerful and interesting data structure in python
# It's called a dataframe
# Let's explore it now
import pandas as pd
purchase_1 = pd.Series({'Name': 'Ali',
                        'Item': 'Item1',
                        'Price': 11.50})
purchase_2 = pd.Series({'Name': 'Hassan',
                        'Item': 'Item2',
                        'Price': 12.50})
purchase_3 = pd.Series({'Name': 'Kamran',
                        'Item': 'Item3',
                        'Price': 15.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()
# How to get a particular row
print(df.loc['Store 2'])
# Get a particular column from a particular row
print(df.loc['Store 1', 'Price'])
#print Price column
print(df['Price'])
# store1's cost
print(df.loc['Store 1']['Price'])
# name and cost of all
print(df.loc[:,['Name', 'Price']])
# Well, if it's a table like structure. Can we query it?
#You guessed it right, we can !!!
print(df['Price'] > 11.5)
print(df.where(df['Price'] > 11.5).dropna())
#there is another way
print(df[df['Price'] > 11.5])
#Remember that or/and operations, let's use them
#Try or first
print(df[(df['Price'] > 11.5) | (df['Name'] == 'Ali')])
#Let's try and now
print(df[(df['Price'] > 11.5) & (df['Name'] == 'Kamran')])
# We can join our tables in database right, what about that in python?
staff_df = pd.DataFrame([{'Name': 'Imran', 'Role': 'Controller Examinations'},
                         {'Name': 'Ali', 'Role': 'Academic Officer'},
                         {'Name': 'Kamran', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'Kamran', 'School': 'CS'},
                           {'Name': 'Asif', 'School': 'EE'},
                           {'Name': 'Ali', 'School': 'BBA'}])
student_df = student_df.set_index('Name')
print(staff_df.head())
print()
print(student_df.head())
#Let's Start with Outer Join
pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
# Inner is cool too, right?
pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)
# Left outer isn't that bad either!!!
pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
# How can I miss your favourite Right outer join :)
pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)
# But there is groupy by clause in SQL right?
# Let's try that too
import numpy as np
df.groupby('Item').agg({'Price': np.average})
# One more thing, here is how we define functions in python
def add(x, y):
    return x + y
add(1, 2)
# Enough from my side now
# Your turn now
# Upvote this notebook now, that's how I will track You have read it
# Fork it, rename it and make following extensions
# 1) Write a function to check a String is palindrome or not
def palindrome():
    x = "racecar"
    y = reversed(x)
    if list(x) == list(y):
        print("It is palindrome")
    else:
        print("It is not palindrome")
        
palindrome()
# 2) write a function check if two String are Anagram?
def anagramSolution(string1,string2):
    alist = list(string2)
    pos = 0
    stillOK = True

    while pos < len(string1) and stillOK:
        posx = 0
        found = False
        while posx < len(alist) and not found:
            if string1[pos] == alist[posx]:
                found = True
            else:
                posx = posx + 1
        if found:
            alist[posx] = None
        else:
            stillOK = False

        pos = pos + 1
    return stillOK
print(anagramSolution('abcd','dcba'))
# 3) write a function which prints first n fibbonaci numbers
def fib(n):
    x=n
    print (0)
    print (1)
    x = x-2
    a = 0
    b = 1
    while(x>0):
        a=a+b
        print (a)
        x = x - 1
        if (x != 0):
            b=b+a
            x=x-1
            print (b)    
fib(8)
# 4) Write a function that finds index of a substring from a string
import re
string = 'Hassaan Ahmed'
substr = re.search(r'\b(Ahmed)\b', string)
print(substr.start())
