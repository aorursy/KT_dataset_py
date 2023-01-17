x=5
x
x,y=(1,2)
x
y
print(x)
print(x,y)
x1=5
type(x1)
x2=5.28
type(x2)
x3=True
type(x3)
#Cannot assign the strings to variables without quatation

x4=Yes

x4
#This is the correct way to assign strings

x4="yes"

x4
type(4.89)
int(3.98)
float(8)
#Object retains the recent value assigned to it. x4 object here outputs "george" and not previously assigned "yes"

x4="george"

x4
#There is no difference in single or double quoted string.

#Both representations can be used interchangeably.

x4='george'

x4
type(x4)
print(x4)
"I'm fine"
'I"m fine'
print(str(x1 ) + " dollar")
print(x1,x2,x3,x4)
3+5 #addition
3-8 #subtraction
3*5 #multiplication
5/3 #division
5**3   #power
5%3    #reminder
y==5**3  #double equality returns boolean operator
y==125
y==2
"friday"[5] #indexing ....In python we count as 0,1,2,... and not as 1,2,3,....
#Function : Indentation is important

def five(x):

    x=5

    return(x)

print(five(3))
#Logical/Boolean operators : 'not,and,or' in the respective preference order.

False or not True and True
#identity operators : is, is not

5 is 6
5==6
5 is not 6
5 !=6
#if statement

if 5!=3*2:

    print("hooray")
#if else statement

if x>3:

    print("case1")

else:

    print("case2")
#elif statement

def compare_to_five(x):

    if x>5:

        print("greater")

    elif x<5:

        print("less")

    else:

        print("equal")
compare_to_five(10)
import pandas as pd

data = [1,2,3,4,5]

series1 = pd.Series(data)

series1
type(series1)
#changing the index of the series object

series1 = pd.Series(data,index=['a','b','c','d','e'])

series1
# Creating a Dataframe using a list

df = pd.DataFrame(data)

df
# Creating a Dataframe using a dictionary

dictionary = {'fruits' :['apple','mango','orange'],'count':[10,20,30]}

df = pd.DataFrame(dictionary)

df
# Creating a Dataframe using a series

series = pd.Series([6,12],index=['a','b'])

series

df = pd.DataFrame(series)

df
# Creating a Dataframe using a Numpy array

import numpy as np

numpyarray = np.array([[50000,60000],['Jane','Jack']])

numpyarray

df = pd.DataFrame({'name':numpyarray[1],'salary':numpyarray[0]})

df