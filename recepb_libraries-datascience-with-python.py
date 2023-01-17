import numpy as np

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#UTF-8 is a variable-width character encoding standard 

#that uses between one and four eight-bit bytes to represent all valid Unicode code points.



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.       
data = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv')

#/kaggle/input/youtube-new/INvideos.csv

#/kaggle/input/youtube-new/CAvideos.csv

#If you run the code above, you need to put the ".csv" path in pd.read_csv () 

data.head()
data.info()
data.corr()
data.head(10)  # The first 10 YouTube Trending Videos
data.views.plot(kind = 'line', color = 'g',label = 'dislikes',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.likes.plot(color = 'b',label = 'likes',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')



data.plot(kind='scatter', x='views', y='comment_count',alpha = 0.5,color = 'red')

plt.xlabel('views')             

plt.ylabel('comment_count')

plt.title('Views - comment_count Scatter Plot')            
# Histogram

# bins = number of bar in figure

data.views.plot(kind = 'hist',bins = 50,figsize = (5,5))

plt.show()
#Program to demonstrate the

#use of user defined functions



def sum(a,b):

   total = a + b 

   return total

x = 10 

y = 20

print("The sum of",x,"and",y,"is:",sum(x, y))
#Local Scope

#A variable created inside a function belongs to the local scope of that function, and can only be used inside that function.

#A variable created inside a function is available inside that function:

def myfunc():

    x = 300

    print(x)

myfunc()
#Function Inside Function

#The variable x is not available outside the function, but it is available for any function inside the function:

#The local variable can be accessed from a function within the function:

def myfunc():

  x = 300

  def myinnerfunc():

    print(x)

  myinnerfunc()



myfunc()
#Global Scope

#A variable created in the main body of the Python code is a global variable and belongs to the global scope.

#Global variables are available from within any scope, global and local.



x = 300

def myfunc():

  print(x)



myfunc()

print(x)
#If you operate with the same variable name inside and outside of a function, Python will treat them as two separate variables, 

#one available in the global scope (outside the function) and one available in the local scope (inside the function):



x = 300

def myfunc():

  x = 200

  print(x)



myfunc()

print(x)
#Global Keyword

#If you need to create a global variable, but are stuck in the local scope, you can use the global keyword.

#The global keyword makes the variable global.

def myfunc():

  global x

  x = 300



myfunc()

print(x)



# How can we learn what is built in scope

#import builtins

#dir(builtins)
def function1(): # outer function

    print ("Hello from outer function")

    def function2(): # inner function

        print ("Hello from inner function")

    function2()



function1()
def function1():          # outer function

    x = 2                 # A variable defined within the outer function

    def function2(a):     # inner function

                          # Let's define a new variable within the inner function rather than changing the value of x of the outer function

        x = 6

        print (a+x)

    print (x)             # to display the value of x of the outer function

    function2(3)



function1()
def student(firstname, lastname ='Balıbey', college ='University of Padua'): 

    print(firstname, lastname, 'studies masters degree at', college, 'in Italy') 

#function student accept one required argument (firstname) and rest two arguments are optional.

student("Recep")
# flexible arguments *args

def f(*args):      # can take many parameters

    for i in args:

        print(i)

f(1,2,3,4)



# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    for key, value in kwargs.items():               

        print(key, ": ", value)

f(country = 'Spain', capital = 'Madrid', population = 6642000)
# Program to show the use of lambda functions

double = lambda x: x * 2

print(double(5))

# Here x is the argument and x * 2 is the expression that gets evaluated and returned.



square = lambda x: x**2     # where x is name of argument

print(square(4))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))
my_list = [1, 5, 4, 6, 8, 11, 3, 12]

new_list = list(filter(lambda x: (x%2 == 0) , my_list))

print(new_list)
my_list = [1, 5, 4, 6, 8, 11, 3, 12]

new_list = list(map(lambda x: x * 2 , my_list))

print(new_list)
# Here is an example of a python inbuilt iterator, value can be anything which can be iterate 

iterable_value = 'Geeks'

iterable_obj = iter(iterable_value) 



while True: 

    try: 

    # Iterate by calling next 

       item = next(iterable_obj) 

       print(item) 

    except StopIteration: 

# exception will happen when iteration will over 

      break

class IterationExample:

    def __iter__(self):

        self.x = 0

        return self



    def __next__(self):

        y = self.x

        self.x += 1

        return y

    

classinstance = IterationExample()

element = iter(classinstance)

# We have created an iterator named element that prints numbers from 0 to N. We first created an instance of the class and 

# we gave it the name classinstance. We then called the iter() built-in method and passed the name of the class instance as the parameter. 

# This creates the iterator object.
class IterationExample:

    def __iter__(self):

        self.x = 0

        return self



    def __next__(self):

        y = self.x

        self.x += 1

        return y



classinstance = IterationExample()

element = iter(classinstance)



print(next(element))

print(next(element))

# We called the next() method and passed the name of the iterator element to the method as the parameter. 

# Every time we do this, the iterator moves to the next element in the sequence. 