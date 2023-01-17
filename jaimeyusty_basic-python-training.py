
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv("../input/insurance.csv")
data.info
data.corr()
data.head(21)
data.columns
data.age.plot(kind = 'line', color = 'g',label = 'age',linewidth=1,alpha = 1,grid = True,linestyle = ':')

data.charges.plot(color = 'r',label = 'charges',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')


plt.legend(loc='upper right')     
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('Line Plot')            
plt.show()




data.plot(kind='scatter', x='age', y='charges',alpha = 0.5,color = 'red')
plt.xlabel('age')              # label = name of label
plt.ylabel('charges')
plt.title('age -- charges') 
data.age.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
dictionary = {'python' : 'def','javascript' : 'var'}
print(dictionary.keys())
print(dictionary.values())
dictionary['java'] = "null"    
print(dictionary)
dictionary['c'] = "int"       
print(dictionary)
del dictionary['java']              
print(dictionary)
print('c' in dictionary)        
dictionary.clear()                   
print(dictionary)
x = data['age']>20    
data[x]
data[np.logical_and(data['age']>20, data['charges']>10000 )]
data[(data['age']>20) & (data['charges']>10000)]
i = 0
while i != 10 :
    print(i)
    i +=1 
print(i)
li = [1,2,3,4,5]
for i in li:
    print(i)
print('')

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(li):
    print(index," : ",value)
print('')   

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary = {'python':'def','javascript':'var'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')

for index,value in data[['age']][0:1].iterrows():
    print(index," : ",value)
def tuple_ex():
     t = (1,2,3)
     return t
a,b,c = tuple_ex()
print(a,b,c)
print(a)
x = 2
def f():
    x = 3
    return x
print(x)
print(f())
x = 3
def f ():
    y = 2*x
    return y
print(f())
def var1():
    
    def var2():
        x = 2
        y = 4
        z = x + y
        return z
    return var2()**2

print(var1())
def f(a,b = 1,c = 3):
    y = a + b + c
    return y
print(f(5))
print(f(5,3,4))
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)

def f(**kwargs):
    for key,value in kwargs.items():
        print(key,"",value)
f(samsun = "good city",istanbul = "bad city")

    
val1 = lambda x: x**2
print(val1(3))
val2 = lambda x,y,z: x+y+z  
print(val2(1,2,3))
numbers = [1,2,3]
x = map(lambda x:x**2,numbers)
print(list(x))
name = "yusuf"
it = iter(name)
print(next(it))
print(*it)
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z = zip(list1,list2)
print(z)
zlist = list(z)
print(zlist)
unzip = zip(*zlist)
unlist,unlist2 = list(unzip)
print(unlist)
print(unlist2)
print(type(unlist))
num1 = [1,2,3]
num2 = [i * 2 for i in num1]
print(num2)
num1 = [5,10,15]
num2 = [i**2 if i == 5  else i + 4  if i > 7 and i <12 else i - 2 for i in num1]
print(num2)

