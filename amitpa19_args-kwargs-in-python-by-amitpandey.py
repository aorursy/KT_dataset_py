# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def myCode(*argv): 

    for arg in argv: 

        print (arg)

        

myCode('Hello', 'Welcome', 'to','the', 'New World') 
def Check_args(f_arg, *argv):

    print("First arg:", f_arg)

    for arg in argv:

        print("Next arg through *argv:", arg)



Check_args('Amit', 'Learning', 'Python', 'Non-Keyword','Argument')
# *args with first extra argument

def myNewCode(arg1, *argv):

    print ("First argument :", arg1)

    for arg in argv:

        print("Next argument through *argv :", arg)

 

myNewCode('Hello', 'Welcome', 'to','the','New World','of','Amit Pandey')
# Will try to clarify with more examples

#Here we will define a function with 2 arguments & use for 2 arguments

def add(x,y):

     print(x + y)



add(5,5)
#Here we will define a function with 2 arguments & use for 3 arguments

#It will throw an error & we need to learn from this error

def add(x,y):

     print(x + y)



add(5,5,2)
#Let's check the result , when we do execute the same with args

def Sum(*args):

     print(sum(args))    



Sum(5,5)

Sum(5,5,6)

Sum(5,5,6,1,2,3,4,5,6,3,4,2,4,5)

def MyCode(*y):

    print(sum(y))

    

MyCode(39,37,41)
#Using args in loop

def MyCode(*args):

    for arg in args:

        if arg/100 > 7:

            print('Very High')

        elif arg/100 >5:

            print('Medium')

        elif arg/100 >0:

            print('Very Low')

MyCode(500,50,890,900,3000,700,80)
#If we want to store the output 

L = []

def MyCode(*args):

    for arg in args:

        if arg/100 > 7:

            flag = 'Very High'

        elif arg/100 >5:

            flag = 'Medium'

        elif arg/100 >0:

            flag = 'Very Low'

        L.append(flag)

        return L

MyCode(500,50,890,900,3000,700,80)
Storing_Output
def Greet_You(**kwargs):

    for key, value in kwargs.items():

        print("{0} = {1}".format(key, value))

              

Greet_You(Hello = "Hello",Name="Friend",How="How",Are="Are",You="You?")

def myCode(**kwargs): 

    for key, value in kwargs.items():

        print ("%s == %s" %(key, value))

 # Driver code

myCode(First ='Believe', Mid ='In', Last='Yourself')   
def myCode(arg1, **kwargs): 

    for key, value in kwargs.items():

        print ("%s == %s" %(key, value))

        

# Driver code

myCode("Hello", First ='Believe', Mid ='In', Last='Yourself')    
def myCode(arg1, arg2, arg3):

    print("arg1:", arg1)

    print("arg2:", arg2)

    print("arg3:", arg3)

     

# Now we can use *args or **kwargs to

# pass arguments to this function : 

args = ("Love", "Your", "Codes")

myCode(*args)

 

kwargs = {"arg1" : "Love", "arg2" : "Your", "arg3" : "Codes"}

myCode(**kwargs)
def myCode(*args,**kwargs):

    print("args: ", args)

    print("kwargs: ", kwargs)

    #Now we can use both *args ,**kwargs to pass arguments to this function

myCode('Love','Your','Codes',First="Love",Mid="Your",Last="Codes")
def calculation(flag='Yes', **i):

    if flag == 'Yes' :

        return sum(i.values())

    else :

        print("Flag is set No")



calculation(x=47, y=24)
calculation(flag='No', x=37, y=92)
#Smart way of using args using lambda

from functools import reduce

def add(*args):

    return reduce(lambda x,y:x+y, args)



add(1,9,20,21,89)