def functionn():

    x=(3,1,5)

    return x

(q,w,e)=functionn()

print(q,w,e)



print('=======================')







def addition(x,y):

    return x+y

def subtraction(x,y):

    return x-y

x=5

y=2

print(addition(x,y),"and",subtraction(x,y))

    
a=5 #it is a global scope

def scopefunk():

    a=15  # it is a local scope beacuse we used in function

    return a

print(a)



# first scope is global but second scope is local scope.



scopefunk()

import builtins

dir(builtins)
def calculatefunction():

    def inside_of_calculatefunction():

        x=4

        y=16

        z=x%y

        return z

    return inside_of_calculatefunction()**4

print(calculatefunction())
def x(a=31,y=1,q=12):

    b=a+y+q

    return b

print(x(1,2,19))

#answer is not 44 beacuse we used different variables.1+2+19=22
def f(*args):

    for i in args:

        print(i)

f(2)

print("=======")

f(1,55,22,11)

print("_______________________")

def funk(**kwargs):

    for key,value in kwargs.items():

        print(key,"model is",value)

funk(BWM="M4",Mercedes="Brabus",Lambogini="Gallardo")
square_root=lambda x:x**1/2

print(square_root(4))



print("=========")

sum=lambda x:x+x

print(sum(123))
citylist=["İstanbul","London","New Delhi"]

y=map(lambda x:x+" city",citylist)

print(list(y))



print("or diffent example")



numberlist=[99,11,23,float(0.54)]

a=map(lambda o:o+12,numberlist)

print(list(a))
example="Donald Trump"

it=iter(example)

print(next(it)) #1

print(next(it)) #2

print(next(it)) #3

print(*it) #and rest of them
list1=[99,88,77,66]

list2=[4,3,2,1]

zipp=zip(list1,list2)

print(zipp)

zipplist=list(zipp)

print(zipplist)
unzip=zip(*zipplist)

newlist1,newlist2=list(unzip)

print(newlist1)

print(newlist2)

print(type(newlist1))
example1=[2,3,4]

result=[i+5*5 for i in example1]

print(result)

#Attention for mathematical priority :)
#Maybe we can make simple exam algorithm.Each question is 10 points.60 is limit for exams

notes=[5,6,9,10,1,0]

notesresult=["Failed" if (i*10)<60 else "Passed" for i in notes]

print(notesresult)

#50(F) 60(P) 90(P) 100(P) 10(F) 0(F)
import numpy as np  

import pandas as pd  

import matplotlib.pyplot as plt

import seaborn as sns

import math

import statistics

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data=pd.read_csv("../input/heart.csv")

data
age_average=(statistics.mean(data["age"]))

print(age_average)
data["Aged_or_Young"]=["Aged" if i>=age_average else "Young" for i in data.age]

data.loc[:,["Aged_or_Young","age"]] 