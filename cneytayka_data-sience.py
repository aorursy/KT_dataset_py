# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

data.info()
data.columns

data.corr()
data.head(10)
plt.style.use('classic') # yazılarımızın estetik durması için matplotun classıc stiilini kullandık.

data.age.plot(kind="line",color="r",label="age",linewidth=1,alpha=1,marker="*",linestyle=":",grid=True,figsize=(15,8))

plt.legend(loc=0)    #best de  yazabilirdik. bu gragik başlıgını (label) ı  uygun yere koyar.

plt.xlabel("index")

plt.ylabel("age")

plt.title("yaş dağılımı")

plt.show()
# bins = number of bar in figure



data.age.plot(kind = 'hist',bins = 75,figsize = (18,12),color="blue")

plt.title("age graph ")

plt.show()
# Scatter Plot

data.plot(kind='scatter', x='trestbps', y='chol',alpha = 0.5,color="red")

plt.xlabel('trestbps')              # label = name of label

plt.ylabel('chol')

plt.title('trestbps chol Scatter Plot')

plt.show()
# data setimizin içinde kaç kişinin şeker hastası oldugunu bulduk.

len(data[data.target==1])

#data[data.chol>200]  tablo seklinde verir

len(data[data.chol>200]) 

data2 = data[data.target==1]

data2
data2.sex.plot(kind="hist",color="green",bins=20)

plt.show()
dictionary={"Galatasaray":"Gs","Fenerbahçe":"Fb","Trabzonspor":"Ts"}

print(dictionary.keys())

print(dictionary.values())
dictionary["Galatasaray"]="r" #update dictionary

print(dictionary)

dictionary["Beşiktaş"]="BJK" #new blok

del dictionary['Galatasaray']

print(dictionary)     #delete

print("Galatasaray" in dictionary) #we check the contents

dictionary.clear() #deleting dictionary

print(dictionary)
series = data['sex']        

print(type(series))



dataframe = data[['sex']] 

print(type(dataframe))
sayac = 0

toplam=0

while sayac<= 10:

    sayac=sayac+ 2

    toplam=toplam+sayac

 

print("0 ile 10 arasındaki çift sayıların toplam:{0}".format(toplam))
# example of what we learn above

def tuble_ex():

    """ return defined t tuble"""

    t = (4,5,6)

    return t

a,b,c = tuble_ex()



print(a,b,c)



print(a,b)



print(a)

# guess print what



x = 5

def f():

    x = 2

    return x

print(x)          # x = 5 global scope

print(f())        # x = 2 local scope
# What if there is no local scope

x = 3

def f():

    k = x**2        # there is no local scope x

    return k

print(f())         # it uses global scope x

# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# How can we learn what is built in scope



import builtins

dir(builtins)
def function1():                             # outer function

    print ("Hello from outer function")

    def function2():                          # inner function

        print ("Hello from inner function")

    function2()



function1()
#rectangular perimeter calculation



def rectangle():

    

    def rectangular_perimeter():

        

        a=5     #long edge

        b=4     #short edge

        A= a + b

        

        return A

    

    return rectangular_perimeter()*2



print(rectangle())

    

    
def function1(name):

    def function2():

        print('Hello ' + name)

    return function2



func = function1('Kio')

func()
dir("args")
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))

# what if we want to change default arguments

print(f(5,4,3))
# default arguments

def a(name,msg="Good Morning"):

    

    print("hello",name,",",msg)



a("Kio")



print("")

# what if we want to change default arguments

a("ahmet","How are you?")
# flexible arguments *args

def f(*args):

    

    for i in args:

        print(i)

        

f("a")

print("")

f(1,2,3,4,5)

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 123456)
#user defined fuctions(long way)

def cube(y): 

    return y*y*y; 

  

g = lambda x: x*x*x 

print(g(7)) 

  

print(cube(5))
# map() with lambda()  

a = [5, 7, 22, 97, 54, 62, 77, 23, 73, 61] 

final_list = list(map(lambda x: x*2 , a)) 

print(final_list)
a = [5, 7, 22, 97, 54, 62, 77, 23, 73, 61] 

final_list = list(filter(lambda x: (x%2 != 0) , a)) 

print(final_list)
from functools import reduce

a = [5, 8, 10, 20, 50, 100] 

sum = reduce((lambda x, y: x + y), a) 

print (sum) 
# iteration example

name = "ronaldo"

it = iter(name)

print(next(it))    # print next iteration

print(*it)         # print remaining iteration

# zip example

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
# Assign integer values to `a` and `b`

a = 4

b = 9



# Create a list with the variables `a` and `b` 

count_list = [1,2,3,a,5,6,7,8,b,10]



print([i for i in count_list])
#example           Q = {x3: x in {0 ... 10}}

#create a list 



Q=[x**3 for x in range(11)]

print(list(Q))

# Conditionals on iterable

num1 = [8,9,10]

num2 = [ i*2 if i/2==0 else i-5  for i in num1]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)



