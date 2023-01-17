import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

data = pd.read_csv('../input/pokemon.csv')
def tuble_ex():

    t = (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
x = 2

def f():

    x = 3

    return x

print(x)      # global variable

print(f())    # function variable
data.columns
def square():

    def add():

        

        x = 2

        y = 3

        z = x + y

        return z

    return add()**add()

print(square())   
def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))

print(f(5,4,3))
def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)

#Dictionary

def f(**kwargs):

    for key, value in kwargs.items():

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 123456)
number_list = [1,2,3]

y=map(lambda x : x**2,number_list)

print(list(y))

##########

def dongu_yazma(*arg):

    return arg[0]**2



z=map(dongu_yazma,number_list)



print(list(z))

############

#pandas apply

import pandas as pd



dictionary = {"NAME":["ali","veli","kenan","hilal","ayse","evren"],

              "AGE":[15,16,17,33,45,66],

              "MAAS": [100,150,240,350,110,220]} 

dataFrame1 = pd.DataFrame(dictionary)

# apply()



def multiply(age):

    return age*2

    

dataFrame1["apply_metodu"] = dataFrame1.AGE.apply(multiply)

print(dataFrame1.apply_metodu)
name = "Ayhan"

it = iter(name)

print(next(it))    # print next iteration

print(next(it))

print(*it)         # print remaining iteration
list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuble

print(un_list1)

print(un_list2)

print(type(un_list2))
num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)
num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
threshold = sum(data.Speed)/len(data.Speed)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]]