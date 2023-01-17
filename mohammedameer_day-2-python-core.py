# Assignment : Addition of 2 numbers
num1 = int(input('Enter 1st number'))
num2 = int(input('Enter 2nd number'))
res = num1+num2
print(res)
type(num1)
'Am'+'eer'
num1 = input('Enter 1st number')
num2 = input('Enter 2nd number')
res = int(num1)+int(num2)
print(res)
# Methods for printing

# Method 1 ( String COncatenation)

print('Sum of '+ num1 + ' and '+num2+' is '+str(res))
# method 2 ( COmma seperator)
print('Sum of',num1,'and',num2,'is',res)
# Method 3 : Formatted String
print(f'Sum of {num1} and {num2} is {res}')
import sys
sys.version
a = ['Ameer',1,1.2,'B']
type(a)
# Advanced Datatype : List[],Tuple(),Dictionary{key:value},Sets{}
a[0] # Addressing the value through index
a[1]
type(a[2])
a[4]
for i in a:
    print(i)
# i is accessing each and every element
a = [1,2,3,4,5] # Input List
b = [1,4,9,16,25] # Output List
a = [1,2,3,4,5]

for i in a:
    print(i**2)
b = []
b
b.append(1.1)
b
# 1. Take the input list as it is
# 2. Take an empty output list
# 3. Take a for loop and append each element value in a new list
list(range(5))
# range is a function taking the values including the first number(0) 
# and excluding last number(upto)
z = []
for i in range(3,31,3):
    z.append(i)
print(z)
a = [10,20,30,40]
b = []
for i in a:
    b.append(i)
print(b)
'A'+'B'
a = ['ameer','Shivraj']
'+'.join(['A','B'])
a = ['aaa','bbb','ccc']
b = []
for i in a: # Here i is not an index, it refers to a specific element
    print(''.join([i,i]))
a = ['aaa','bbb','ccc']
b = []
length = len(a)
#print(length)
for i in range(len(a)): # Here i is an index of a list
    print(''.join([a[i],a[i]]))
# Function

# 1. Function Definition
# 2 .Function Call
def res(a,b):
   print(a+b)
   print(a-b)
   print(a*b)
   print(a/b)
res(2,3)
res(10,3)
res(2,4)
def amr():
    z = []
    for i in range(3,31,3):
        z.append(i)
    print(z)
amr()
a = ['ameer','Shivraj']
''.join(a)
a[1]
a= ['bhoj','vilas']
' '.join([a[0],a[1]])
# Basic Libraries for Data Science

# Numpy : Numerical Computing
# Matplotlib : Data Visualization (Line Plot, Scatter Plot, Bar Plot, Histogram, Pie Chart)
# Pandas : Data Analysis
import numpy as np
# Numpy Arrays : Scalar 0 D, Vector - 1D, Matrix - 2D

# Scalar
a = np.array(34)
print(a)
print(a.ndim)
# Vector
a = np.array([1,2,3,4])
print(a)
print(a.ndim)
# Matrix
a = np.array([[1,2,3,4],[2,3,4,5]])
print(a)
print(a.ndim)
import pandas as pd
df = pd.DataFrame(a)
df
np.mean(a)
np.median(a)
np.max(a)
np.min(a)
a = np.array([1,2,3,4,5,4,4])
a
from scipy import stats
stats.mode(a)
# Normalization : Scaling to one specific range of values (0-1)
xin = np.array([1,2,3,4,5,6,7,8,9,10])
xin
xmin = np.min(xin)
xmax = np.max(xin)
print(xmin)
print(xmax)
xnorm = (x-xmin)/(xmax-xmin)
xnorm
import matplotlib.pyplot as plt

# import package.subpackage(libray) as nickname
# Library - Set of code
# Package - COllection of multiple libraries
name = ['Ameer','Ketan','Shivraj','Vatsal']
age = [26,18,19,20]
c = ['r','#e39529','#62a371','#0349fc']
plt.bar(name,age,width = 1.0,color = c)
plt.xlabel('Names of people')
plt.ylabel('Age of people')
plt.title('My first Bar Graph in Python')
plt.show()

name = ['Ameer','Ketan','Shivraj','Vatsal']
weight = [56,45,53,46]
height = [113,116,124,108]
index = np.array([0,1,2,3])
plt.bar(name,height,width = 0.4)
plt.bar(index+0.4,weight,width = 0.4)
plt.xlabel('Names of people')
plt.ylabel('Age of people')
plt.title('My first Bar Graph in Python')
plt.show()
