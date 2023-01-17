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
#Python-1-program
#Calculator
print("The options available are:")
print("1.+\n2.-\n3.*\n4./\n5.%\n6.//")
print("Enter your choice:")
a=int(input())
if a==1:
    print("Enter the 2 numbers:")
    b=float(input())
    c=float(input())
    d=b+c
    print("The sum of two number is:",d)
elif a==2:
    print("Enter the 2 numbers:")
    b=float(input())
    c=float(input())
    d=abs(b-c)
    print("The difference between  numbers is:",d)
elif a==3:
    print("Enter the 2 numbers:")
    b=float(input())
    c=float(input())
    d=b*c
    print("The product of two number is:",d)
elif a==4:
    print("Enter the dividend:")
    b=float(input())
    print("Enter the divisor:")
    c=float(input())
    d=b/c
    print("The division of b and c is:",d)
elif a==5:
    print("Enter the number to be divided:")
    b=float(input())
    print("Enter the number by which modulus must be done:")
    c=float(input())
    d=b%c
    print("the modulus is:",d)
elif a==6:
    print("Enter the dividend:")
    b=float(input())
    print("Enter the divisor:")
    c=float(input())
    d=b//c
    print("The division(flooring) of b and c is:",d)
#Python-2-program
#to check prime && (even or odd)
print("This programs is to check whether a number is prime,even or odd")
print("Enter a number:")
a=int(input())
c=0
p=3
for i in range(1,a+1):
    r=a%i
    if r==0:
        c+=1

if c>2:
    p=0
elif c==2:
    p=1
if a%2==0:
    e=1    
elif a%2!=0:
     e=0 
if p==0 and e==0:
    print("Number is not Prime and an odd number")
elif p==1 and e==0:
    print("Number is prime and an odd number")
elif p==0 and e==1:
    print("Number is not prime and an even number")
elif p==1 and e==1:
    print("Number is prime and an even number")
#Python-3-program
#to find the roots of the quadratic equation
import cmath

a = int(input())
b = int(input())
c = int(input())

# calculate the discriminant
d = (b**2) - (4*a*c)

# find two solutions
sol1 = (-b-cmath.sqrt(d))/(2*a)
sol2 = (-b+cmath.sqrt(d))/(2*a)

print('The solution are {0} and {1}'.format(sol1,sol2))
#Python-4-program
#Sets and their operatoins
# set of integers
a = {1, 2, 3}
print(a)
# set of mixed datatypes
b = {1.0, "Hello", (1, 2, 3)}
print(b)
# add an element
a.add(6013)
print(a)
# add multiple elements
a.update([2018, 6013, 5060])
print(a)
# discard an element
a.discard(5060)
print(a)
# remove an element
a.remove(2018)
print(a)
# pop an element
# Output: random element
print(b)
print(b.pop())
print(b)
# clear b
# Output: set()
b.clear()
print(b)
c={1,2,3,4,5,6}
d={4,5,6,7,8,9}
# Set union method
print(c|d)
#intersection of sets
print(c&d)
#difference of 2 sets
print(c-d)
#symmetric difference between sets
print(c^d)
#Python-5-program
#bubble sort
arr = [6,0,1,3,1,8];     
t = 0;  
print("Elements of original array: ");    
for i in range(0, len(arr)):    
    print(arr[i], end=" ");    
for i in range(0, len(arr)):    
    for j in range(i+1, len(arr)):    
        if(arr[i] > arr[j]):    
            t = arr[i];    
            arr[i] = arr[j];    
            arr[j] = t;    
     
print();
print("Elements of array sorted in ascending order: ");    
for i in range(0, len(arr)):    
    print(arr[i], end=" "); 
#python-6-program
#Narcissist number(amstrong number)
a=int(input("Enter an number:"))
t=a
t1=a
c=0
s=0
while a>0:
    a=a//10
    c+=1
while t>0:
    r=t%10
    t//=10
    s=s+pow(r,c)
if t1 == s:
    print("Given number is narcissist.")
else:
    print("The number is not narcissist.")
#python-7-program
#constructors
#declaring variables in the class 
class si:
    p=10500
    t=3.5
    r=0.99
    def st(self):
        print('The given values are:')
        print(self.p)
        print(self.t)
        print(self.r)
        print('The simple interest is:')
        print((self.p*self.t*self.r)/100)
a=si()
a.st()
#creating a class with a constructor
class bmi:
    def __init__(self,w,h):
        self.w=w
        self.h=h
    def disp(self):
        print('The given weigth is:',self.w)
        print('the given heigth is:',self.h)
        c=((self.w*703)/(self.h*self.h))
        print('BMI:',c)
a=bmi(95,171)
a.disp()
#creating more than one constructor in a class
class con:
    def __init__(self):
        print('This is the first constructor')
    def __init__(self):
        print('This is second constructor')
a=con()
#Python-8-program
#palindrome_string
a =input()
# make it suitable for caseless comparison
a = a.casefold()
# reverse the string
b = reversed(a)
# check if the string is equal to its reverse
if list(a) == list(b):
   print("The string is a palindrome.")
else:
   print("The string is not a palindrome.")
#Python-9-program
#generators in python
#basic generator function
def igen():
    n=1
    if n%2==0:
        print('This is an odd number')
    else:
        print('This is an even number')
    yield n
    n=n+1
    if n%2==0:
        print('This is an odd number')
    else:
        print('This is an even number')
    yield n
    n+=1
    if n%2==0:
        print('This is an odd number')
    else:
        print('This is an even number')
    yield n
a= igen()
next(a)
print(next(a))
#using a for loop
for i in igen():
    print(i)
print("\n")
print("second snipet:")
print("\n")
#to find the factorial of the given number
def fact(n):
    a=1
    c=1
    while c<=n:
        if c==n:
            yield a
        a=a*c
        c=c+1
b=fact(5)
for i in fact(7-1):
    print(i)
#Python-10-program
#program to merge two text files
import shutil 
from pathlib import Path 
   
firstfile = Path(r'../input/textfilestomerge/intro.txt') 
secondfile = Path(r'../input/textfilestomerge/now.txt') 
  
newfile = input("Enter the name of the new file: ") 
print() 
print("The merged content of the 2 files will be in", newfile) 
  
with open(newfile, "wb") as wfd: 
  
    for f in [firstfile, secondfile]: 
        with open(f, "rb") as fd: 
            shutil.copyfileobj(fd, wfd, 1024 * 1024 * 10) 
  
print("\nThe content is merged successfully.!") 
print("Do you want to view it ? (y / n): ") 
  
check = input() 
if check == 'n': 
    exit() 
else: 
    print() 
    c = open(newfile, "r") 
    print(c.read()) 
    c.close()
print("\n\n")
print("***********Pandas Library**********")
print("\n\n")
#Pandas(1-1)
#reading and writing in a csv file using pandas library
import pandas as pd
df = pd.read_csv (r'../input/commoditiescsv/commodities.csv',index_col=0)
print (df)
#now writing new data to the csv file(old contents will be deleted)
df = pd.DataFrame([['apple', 1, 100], ['banana', 1, 50]], columns = ['Item', 'Quantity(kg)','Cost'])
df.to_csv('commodities.csv')
print(df)
# Print out cost column as Pandas Series(new data)
print(df['Cost'])
# Print out cost column as Pandas DataFrame(new data)
print(df[['Cost']])

#Pandas(1-2)
#operators in retrieving data
import pandas as pd
df = pd.read_csv (r'../input/commodities1/commodities1.csv',index_col=0)
#using equals operator
print("\n== operator")
print(df.loc[df['Cost'] == 50])
#using greater-than operator
print("\n< operator")
print(df.loc[df['Cost'] > 30])
#using less-than operator
print(df.loc[df['Cost'] < 60])
#using not-equal-to operator
print("\n!= operator")
print(df.loc[df['Item'] != 'Daal'])
#multiple operator to retrieve a particular data
print("\nMultiple Conditions")
print(df.loc[(df['Item'] != 'Rice') & (df['Item'] == 'brinjal')])
#Pandas(1-3)
#Windows function
#.rolling() Function
print("\nrolling function")
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('6/10/2020', periods=10),
   columns = ['A', 'B', 'C', 'D'])
print (df.rolling(window=3).mean())
#.expanding() Function
print("\nexpanding function")
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A', 'B', 'C', 'D'])
print(df.expanding(min_periods=3).mean())
#.ewm() Function
print("\newm function")
import pandas as pd
import numpy as np
 
df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A', 'B', 'C', 'D'])
print (df.ewm(com=0.5).mean())
#Pandas(1-4)
#converting of datatypes
#retireving data from csv file
import pandas as pd
df = pd.read_csv (r'../input/commodities2/commodities2.csv',index_col=0)
print(df)
#changing int64 to string
print("before changing the datatype")
print(df.dtypes)
df['Cost'] = df['Cost'].astype(str)
print("After changing the int64 to string")
print(df.dtypes)
print("\n")
#coverting float64 to int32
print("float64 to int32")
df['Quantity(kg)'] = df['Quantity(kg)'].astype(int)
print(df.dtypes)
#Pandas(1-5)
#row operations i.e inserting
#retireving data from csv file
import pandas as pd
df = pd.read_csv (r'../input/commodities2/commodities2.csv',index_col=0)
print("\n------------ BEFORE ----------------\n")
print(df)
#Adding row at the end
df.loc[len(df)] = ['pickle',5, 550] 
print("\n------------ AFTER ----------------\n")
print(df)
print("\n")
#Add row with specific index name
print("Adding row with specific index name to the data")
print("\n------------ BEFORE ----------------\n")
print(df)
df.loc['15'] = ['chicken', 2, 360]
print("\n------------ AFTER ----------------\n")
print(df)
print("here the desired index position is 15\n")
#Pandas(1-6)
#concat,combine_first,append
import pandas as pd
a = {'A': 6031, 'B': 6071}
b = {'B': 6103, 'C': 6096, 'D': 6010}
df1 = pd.DataFrame(a, index=[0])
df2 = pd.DataFrame(b, index=[1])
d1 = pd.DataFrame()
d2 = pd.concat([df1, df2]).fillna(0)
print("\n------------ concat ----------------\n")
print(d2)
d1 = d1.append(df1)
d1 = d1.append(df2).fillna(0)
print("\n------------ append ----------------\n")
print(d1)
d3 = pd.DataFrame()
d3 = d3.combine_first(df1).combine_first(df2).fillna(0)
print("\n------------ combine_first ----------------\n")
print(d3)
#Pandas(1-7)
#mean,mode,median,covariance
import pandas as pd
df = pd.read_csv (r'../input/commodities2/commodities2.csv',index_col=0)
print("\nTo find the mean of the numericis in the dataset\n")
print(df.mean())
print("\nTo find the mode of the numericis in the dataset\n")
print(df.mode())
print("\nTo find the median of the numericis in the dataset\n")
print(df.median())
print("\nTo find the covariance of the numericis in the dataset\n")
print(df.cov())
#Pandas(1-8)
#function application
#axis prameter
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.apply(np.mean,axis=1)
print (df.apply(np.mean))
print("\nElement wise function application")
#element-wise
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
# My custom function
df['col1'].map(lambda x:x*100)
print(df.apply(np.mean))
#Table-wise Function Application
print("\nTable wise function application")
import pandas as pd
import numpy as np
def adder(ele1,ele2):
   return ele1+ele2
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.pipe(adder,2)
print (df.apply(np.mean))
#Row or Column Wise Function Application
print("\nrow or column wise function application")
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])
df.apply(np.mean,axis=1)
print (df.apply(np.mean))
#Pandas(1-9)
#filtering
#retireving data from csv file
import pandas as pd
df = pd.read_csv (r'../input/commodities2/commodities2.csv',index_col=0)
print(df)
print("\nFilter rows which contain specific keyword")
print("\n---- Filter with State contains Daal ----\n")
df1 = df[df['Item'].str.contains("Oil")]
print(df1)
#Filtering DataFrame Index
print("based on the string in the index data retireval")
print("\n---- Filter Index contains al ----\n")
df.Item = df.Item.astype('str')
df2 = df[df.Item.str.contains('al')]
print(df2)
#Filtering using AND operator
print("\n---- Filter DataFrame using & ----\n")
df.Item = df.Item.astype('str')
df1 = df[df.Item.str.contains('al') & df.Item.str.contains('a')]
print(df1)
#Pandas(1-10)
#Aggregation
#Aggregations on DataFrame
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('1/6/2020', periods=10),
   columns = ['A', 'B', 'C', 'D'])
print (df)
r = df.rolling(window=3,min_periods=1)
print (r)
#Apply Aggregation on a Whole Dataframe
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10, 4),
   index = pd.date_range('1/6/2020', periods=10),
   columns = ['A', 'B', 'C', 'D'])
print (df)
r = df.rolling(window=3,min_periods=1)
print (r.aggregate(np.sum))
print("\n")
print("\n***********Numpy Library***********")
print("\n")
#numpy(2-1)
#various check methods for the elements in an array
# to check whether zero is present or not
import numpy as np
x = np.array([10, 25, 36, 48])
print("Original array:")
print(x)
print("Test if any elements is zero:")
print(np.all(x))
x = np.array([0, 100, 200, 300])
print("Original array:")
print(x)
print("Test if any elements is zero:")
print(np.all(x))
#for checking the element if it is infinity
print("\nTo check the +ve,-ve, infinity elements:")
import numpy as np
a = np.array([6, 0, 1, 3, np.inf])
print("Original array")
print(a)
print("Test element-wise for infinity:(if +ve or -ve the output is False)")
print(np.isinf(a))
#for checking the nan-elements in the array
print("\nTo check if nan elements are present in array:")
import numpy as np
a = np.array([1, 0, np.nan, np.inf])
print("Original array")
print(a)
print("Test element-wise for NaN:(if int,infinity elements then output is false)")
print(np.isnan(a))
#check for real,complex and scalar data
print("\nTo check if the real,complex and scalar elements are there in data:")
import numpy as np
a = np.array([60+13j, 50+60j, 6013, 6071.103, 2, 6103j])
print("Original array")
print(a)
print("Checking for complex number:")
print(np.iscomplex(a))
print("Checking for real number:")
print(np.isreal(a))
print("Checking for scalar type:")
print(np.isscalar(3.1))
print(np.isscalar([3.1]))
#numpy(2-2)
#comparing 2 similar arrays using operators
import numpy as np
x = np.array([6013, 6071, 6103])
y = np.array([6002, 6010, 6096])
print("Original numbers:")
print(x)
print(y)
print("Comparison - greater(>)")
print(np.greater(x, y))
print("Comparison - greater_equal(>=)")
print(np.greater_equal(x, y))
print("Comparison - less(<)")
print(np.less(x, y))
print("Comparison - less_equal(<=)")
print(np.less_equal(x, y))
#size determination i.e the space occupied by the elements
print("\nFinding the size ocuupied by the elements:\n")
print("Size of the memory occupied by the first(x) array:")
print("%d bytes" % (x.size * x.itemsize))
print("Size of the memory occupied by the second(y) array:")
print("%d bytes" % (y.size * y.itemsize))
#numpy(2-3)
#saving an array to a pre-existing text file
import numpy as np
import os
x = np.arange(12).reshape(4, 3)
print("Original array:")
print(x)
header = 'col1 col2 col3'
np.savetxt('text-arr.txt', x, fmt="%d", header=header) 
print("After loading, content of the text file:")
result = np.loadtxt('text-arr.txt')
print(result)
#numpy(2-4)
#finding the unique elements in n array
import numpy as np
x = np.array([6013, 6071, 6103, 6096, 6010, 6013])
print("Original array:")
print(x)
print("Unique elements of the above array:")
print(np.unique(x))
#finding the unique elements in a multi-dimensional array
x = np.array([[1, 1], [2, 3]])
print("Original array:")
print(x)
print("Unique elements of the above array:")
print(np.unique(x))
#common values
print("\nFinding the common values in the 2 arrays")
import numpy as np
array1 = np.array([0, 10, 20, 40, 60])
print("Array1: ",array1)
array2 = [10, 30, 40]
print("Array2: ",array2)
print("Common values between two arrays:")
print(np.intersect1d(array1, array2))
#numpy(2-5)
#multiplication,determinant,Einstein’s summation
#multiplication of 2 square matrix
import numpy as np
p = [[1, 2], [3, 4]]
q = [[8, 7], [6, 5]]
print("original matrix:")
print(p)
print(q)
result1 = np.dot(p, q)
print("Result of the matrix multiplication:")
print(result1)
#determinant of a square matrix
print("\nDeterminant of a square matrix:")
import numpy as np
from numpy import linalg as LA
a = np.array([[1, 2], [3, 4]])
print("Original 2-d array")
print(a)
print("Determinant of the said 2-D array:")
print(np.linalg.det(a))
#Einstein summation
print("\nEinstein's Summation over an array:-")
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
print("Original 1-d arrays:")
print(a)
print(b)
result =  np.einsum("n,n", a, b)
print("Einstein’s summation convention of the said arrays:")
print(result)
x = np.arange(9).reshape(3, 3)
y = np.arange(3, 12).reshape(3, 3)
print("Original Higher dimension:")
print(x)
print(y)
result = np.einsum("mk,kn", x, y)
print("Einstein’s summation convention of the said arrays:")
print(result)
#numpy(2-6)
#sorting complex and particular range of data
#sorting the complex numbers
import numpy as np
complex_num = [1 + 2j, 3 - 1j, 3 - 2j, 4 - 3j, 3 + 5j]
print("Original array:")
print(complex_num)
print("\nSorted a given complex array using the real part first, then the imaginary part.")
print(np.sort_complex(complex_num))
#sorting particular no of elements from the top of the array
print("\nSorting only the top 5 elements of the array:")
import numpy as np
nums =  np.random.rand(10)
print("Original array:")
print(nums)
print("\nSorted first 5 elements:")
print(nums[np.argpartition(nums,range(5))])
#numpy(2-7)
#mathematical functions and operations
#mathematical operations using numpy
import numpy as np
print("Add:")
print(np.add(6013, 5060))
print("Subtract:")
print(np.subtract(6013, 5060))
print("Multiply:")
print(np.multiply(6013, 5060))
print("Divide:")
print(np.divide(6013, 5060))
#exponentiation
print("\nExponentiation:")
import numpy as np
l1 = np.log(1e-50)
l2 = np.log(2.5e-50)
print("Logarithm of the sum of exponentiations:")
print(np.logaddexp(l1, l2))
print("Logarithm of the sum of exponentiations of the inputs in base-2:")
print(np.logaddexp2(l1, l2))
#abs() function
print("\nabsolute function")
import numpy as np
x = np.array([-10.2, 122.2, .20])
print("Original array:")
print(x)
print("Element-wise absolute value:")
print(np.absolute(x))
#flooring,ceiling,truncation
print("\nflooring,ceiling,truncation")
import numpy as np
x = np.array([-60.13, -60.71, -61.03, 60.96, 60.10, 6002, 6082])
print("Original array:")
print(x)
print("Floor values of the above array elements:")
print(np.floor(x))
print("Ceil values of the above array elements:")
print(np.ceil(x))
print("Truncated values of the above array elements:")
print(np.trunc(x))
#numpy(2-8)
#String operations
#encoding and decoding
print("\nTo encode and decode the data of array of strings")
import numpy as np
x = np.array(['python exercises', 'PHP', 'java', 'C++'], dtype=np.str)
print("Original Array:")
print(x)
encoded_char = np.char.encode(x, 'cp500')
decoded_char = np.char.decode(encoded_char,'cp500')
print("\nencoded =", encoded_char)
print("decoded =", decoded_char)
#removing trailing whitespaces
print("\nTo remove the trailing whitespaces in the array of strings:")
import numpy as np
x = np.array([' python exercises ', ' PHP  ', ' java  ', '  C++'], dtype=np.str)
print("Original Array:")
print(x)
rstripped_char = np.char.rstrip(x)
print("\nRemove the trailing whitespaces : ", rstripped_char)
#logical operators for array of strings
import numpy as np
x1 = np.array(['Hello', 'PHP', 'JS', 'examples', 'html'], dtype=np.str)
x2 = np.array(['Hello', 'php', 'Java', 'examples', 'html'], dtype=np.str)
print("\nArray1:")
print(x1)
print("Array2:")
print(x2)
print("\nEqual test:")
r = np.char.equal(x1, x2)
print(r)
print("\nNot equal test:")
r = np.char.not_equal(x1, x2)
print(r)
print("\nLess equal test:")
r = np.char.less_equal(x1, x2)
print(r)
print("\nGreater equal test:")
r = np.char.greater_equal(x1, x2)
print(r)
print("\nLess test:")
r = np.char.less(x1, x2)
print(r)
#numpy(2-9)
#polynomial root,commutation,operations
#finding the roots of the polynomial
print("\nroots of the \nx2-2x+1\nx4-12x3+10x2+7x=10\nx2-5x+6")
import numpy as np
print("Roots of the first polynomial:")
print(np.roots([1, -2, 1]))
print("Roots of the second polynomial:")
print(np.roots([1, -12, 10, 7, -10]))
print("Roots of the third polynomial:")
print(np.roots([1,-5,6]))
#computing the values of the polynomial
print("\nto find the polynomial computed value:")
import numpy as np
print("Polynomial value when x = 2:")
print(np.polyval([1, -2, 1], 2))
print("Polynomial value when x = 3:")
print(np.polyval([1, -12, 10, 7, -10], 3))
#polynomial operations
print("\npolynomial add,sub,mult,div")
from numpy.polynomial import polynomial as P
x = (10,20,30)
y = (30,40,50)
print("Add one polynomial to another:")
print(P.polyadd(x,y))
print("Subtract one polynomial from another:")
print(P.polysub(x,y))
print("Multiply one polynomial by another:")
print(P.polymul(x,y))
print("Divide one polynomial by another:")
print(P.polydiv(x,y))
#numpy(2-10)
#trignometric operations
#computing sine,cosine,tangent 
print("\nTo print the trignometric values for the given angles")
import numpy as np
print("sine: array of angles given in degrees")
print(np.sin(np.array((0., 30., 45., 60., 90.)) * np.pi / 180.))
print("cosine: array of angles given in degrees")
print(np.cos(np.array((0., 30., 45., 60., 90.)) * np.pi / 180.))
print("tangent: array of angles given in degrees")
print(np.tan(np.array((0., 30., 45., 60., 90.)) * np.pi / 180.))
#computing inverse sine,cosine,tangent
print("\nTo print the inserve trignometric vales of the given elements:")
import numpy as np
x = np.array([-1., 0, 1.])
print("Inverse sine:", np.arcsin(x))
print("Inverse cosine:", np.arccos(x))
print("Inverse tangent:", np.arctan(x))
#radians to degrees
print("\nTo covert the radians to degrees:")
import numpy as np
x = np.array([-np.pi, -np.pi/2, np.pi/2, np.pi])
r1 = np.degrees(x)
r2 = np.rad2deg(x)
assert np.array_equiv(r1, r2)
print(r1)
#degrees to radians
print("\nTo convert degrees to radians")
import numpy as np
x = np.array([-180.,  -90.,   90.,  180.])
r1 = np.radians(x)
r2 = np.deg2rad(x)
assert np.array_equiv(r1, r2)
print(r1)
#hyperbolic values
print("\nTo print hyperbolic values of the given elements")
import numpy as np
x = np.array([-1., 0, 1.])
print(np.sinh(x))
print(np.cosh(x))
print(np.tanh(x))