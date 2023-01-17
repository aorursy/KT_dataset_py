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
                                                            #Pandas library programs

#1.1-reading dictionary



#reading dictionary statically

dict = {"subjects": ["Os", "DBMS", "Unix", "Networks", "Oops"],

       "credits": ["3", "3", "3", "4", "3"],

       "assgn": ["scheduling algo", "databse", "kernel ops", "packet switching", "inheritance"],

       "time": [40, 40, 40, 60, 40] }



import pandas as pd

a = pd.DataFrame(dict)

print(a)

print("After indexing:")

a.index=[1,2,3,4,5]

print(a)
#1.2-reading in series and dataframe



import pandas as pd

df = pd.read_csv ('../input/commoditiescsv/commodities.csv',index_col=0)

print (df)

# Print out cost column as Pandas Series

print(df['Cost'])

# Print out cost column as Pandas DataFrame

print(df[['Cost']])

# Print out DataFrame with Item and cost columns

print(df[['Item', 'Cost']])
#1.3-indexing of data



# Import commodities data data

import pandas as pd

df = pd.read_csv('../input/commoditiescsv/commodities.csv', index_col = 0)

# Print out first 4 observations

print(df[0:4])

# Print out fifth and sixth observation

print(df[4:6])

#print till the required number

print(df[:6])

# Print out observation for 3rd element

print(df.iloc[2])

# Print out observations for rice and sugar

print(df.loc[[1, 10]])
#1.4-Series operations



#Create a Series from ndarray

#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

data = np.array(['Os','Dbms','Networks','Compiler eng'])

s = pd.Series(data)

print(s)

print("\n")

print("secondnd snipet:")

print("\n")

#indexing

import pandas as pd

import numpy as np

data = np.array(['Os','Dbms','Networks','Compiler eng'])

s = pd.Series(data,index=[1,2,3,4])

print(s)

print("\n")

print("third snipet:")

print("\n")

#Create a Series from Scalar

#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

s = pd.Series(6013, index=[1,2,3,4])

print(s)

print("\n")

print("fourth snipet:")

print("\n")

import pandas as pd

s = pd.Series([6013,2018,5060,13,5013],index = ['a','b','c','d','e'])

#retrieve the first element

print(s[0])

#indexslicing

print(s[:3])

#negative indexing

print(s[-3:])

#retrieval using label

print(s[:'c'])

#retrieval multiple items

print(s['a':'d'])





#1.5-Dataframe opeartions



print("first snipet:")

#creating dataframe with a list

import pandas as pd

data = [6013,2018,5060,13,5013]

df = pd.DataFrame(data)

print(df)

print("\n")

print("second snipet:")

print("\n")

import pandas as pd

data = [['Akula Sainadh',6013,20],['Naveen',6071,20],['Sakthi',6103,20]]

df = pd.DataFrame(data,columns=['Name','Id','Age'])

print(df)

print("\n")

print("third snipet:")

print("\n")

#changing of datatype

import pandas as pd

data = [['Akula Sainadh',6013,20],['Naveen',6071,20],['Sakthi',6103,20]]

df = pd.DataFrame(data,columns=['Name','Id','Age'],dtype='float')

print(df)

print("\n")

print("fourth snipet:")

print("\n")

#creating an indexed DataFrame using arrays

import pandas as pd

data = {'Name':['Sainadh', 'Naveen', 'Sakthi', 'Rizwan'],'Age':[20,20,20,20]}

df = pd.DataFrame(data, index=['1','2','3','4'])

print(df)

print("\n")

print("fifth snipet:")

print("\n")

#Create a DataFrame from List of Dicts

import pandas as pd

data = [{'Sainadh': 6013, 'Naveen': 6071},{'Sakthi': 6103, 'Rizwan': 6096, 'Ajai': 6010}]

df = pd.DataFrame(data)

print(df)

print("\n")

print("sixth snipet:")

print("\n")

#column addition

import pandas as pd



d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),

   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}



df = pd.DataFrame(d)



# Adding a new column to an existing DataFrame object with column label by passing new series



print ("Adding a new column by passing as Series:")

df['three']=pd.Series([10,20,30],index=['a','b','c'])

print(df)



print ("Adding a new column using the existing columns in DataFrame:")

df['four']=df['one']+df['three']



print(df)

print("\n")

print("seventh snipet:")

print("\n")

#column deletion

import pandas as pd



d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 

   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']), 

   'three' : pd.Series([10,20,30], index=['a','b','c'])}



df = pd.DataFrame(d)

print ("Our dataframe is:")

print(df)



# using del function

print ("Deleting the first column using DEL function:")

del df['one']

print(df)



# using pop function

print ("Deleting another column using POP function:")

df.pop('two')

print(df)
#1.6- Pandas Basic functions



print("first snipet:")

import pandas as pd

import numpy as np

s = pd.Series(np.random.randn(4))

print ("The axes are:")

print(s.axes)

print("\n")

print("second snipet:")

print("\n")

#no of dimensions(ndim)

import pandas as pd

import numpy as np

#Create a series with 4 random numbers

s = pd.Series(np.random.randn(4))

print(s)

print ("The dimensions of the object:")

print(s.ndim)

print("\n")

print("third snipet:")

print("\n")

#size of the series

import pandas as pd

import numpy as np

#Create a series with 4 random numbers

s = pd.Series(np.random.randn(2))

print(s)

print ("The size of the object:")

print(s.size)

print("\n")

print("fourth snipet:")

print("\n")

#values

import pandas as pd

import numpy as np

#Create a series with 4 random numbers

s = pd.Series(np.random.randn(4))

print(s)

print ("The actual data series is:")

print(s.values)

print("\n")

print("fifth snipet:")

print("\n")

#head

import pandas as pd

import numpy as np

#Create a series with 4 random numbers

s = pd.Series(np.random.randn(4))

print ("The original series is:")

print(s)

print ("The first tree rows of the data series:")

print(s.head(3))

print("\n")

print("sixth snipet:")

print("\n")

#tail

import pandas as pd

import numpy as np



#Create a series with 4 random numbers

s = pd.Series(np.random.randn(4))

print ("The original series is:")

print(s)

print ("The last three rows of the data series:")

print(s.tail(3))
#1.7- Pandas Statistics Functions

print("first snipet")

#data for doing the statistics operations

import pandas as pd

import numpy as np

#Create a Dictionary of series

d = {'Name':pd.Series(['sai','nadh','saiman','saiman2k','saiman2kak','akula','sainadh akula',

   'manju','nadh','akula manju','saiman2ka3k','manjunadh akula']),

   'Age':pd.Series([18,20,21,19,20,10,22,18,19,22,21,20]),

   'Rating':pd.Series([9.73,6.84,5.98,8.56,7.20,9.6,5.8,8.78,9.98,6.80,8.10,9.65])

}

#Create a DataFrame

df = pd.DataFrame(d)

print(df)

print("\n")

print("second snipet:")

print("\n")

#sum

import pandas as pd

import numpy as np

#Create a Dictionary of series

d = {'Name':pd.Series(['sai','nadh','saiman','saiman2k','saiman2kak','akula','sainadh akula',

   'manju','nadh','akula manju','saiman2ka3k','manjunadh akula']),

   'Age':pd.Series([18,20,21,19,20,10,22,18,19,22,21,20]),

   'Rating':pd.Series([9.73,6.84,5.98,8.56,7.20,9.6,5.8,8.78,9.98,6.80,8.10,9.65])

}

#Create a DataFrame

df = pd.DataFrame(d)

print(df.sum())

print("\n")

print("third snipet:")

print("\n")

#axis(particular)

import pandas as pd

import numpy as np

#Create a Dictionary of series

d = {'Name':pd.Series(['sai','nadh','saiman','saiman2k','saiman2kak','akula','sainadh akula',

   'manju','nadh','akula manju','saiman2ka3k','manjunadh akula']),

   'Age':pd.Series([18,20,21,19,20,10,22,18,19,22,21,20]),

   'Rating':pd.Series([9.73,6.84,5.98,8.56,7.20,9.6,5.8,8.78,9.98,6.80,8.10,9.65])

}

#Create a DataFrame

df = pd.DataFrame(d)

print(df.sum(1))

print("\n")

print("fourth snipet:")

print("\n")

#mean()

import pandas as pd

import numpy as np

#Create a Dictionary of series

d = {'Name':pd.Series(['sai','nadh','saiman','saiman2k','saiman2kak','akula','sainadh akula',

   'manju','nadh','akula manju','saiman2ka3k','manjunadh akula']),

   'Age':pd.Series([18,20,21,19,20,10,22,18,19,22,21,20]),

   'Rating':pd.Series([9.73,6.84,5.98,8.56,7.20,9.6,5.8,8.78,9.98,6.80,8.10,9.65])

}

#Create a DataFrame

df = pd.DataFrame(d)

print(df.mean())

print("\n")

print("fifth snipet:")

print("\n")

#stndard deviation

import pandas as pd

import numpy as np

#Create a Dictionary of series

d = {'Name':pd.Series(['sai','nadh','saiman','saiman2k','saiman2kak','akula','sainadh akula',

   'manju','nadh','akula manju','saiman2ka3k','manjunadh akula']),

   'Age':pd.Series([18,20,21,19,20,10,22,18,19,22,21,20]),

   'Rating':pd.Series([9.73,6.84,5.98,8.56,7.20,9.6,5.8,8.78,9.98,6.80,8.10,9.65])

}

#Create a DataFrame

df = pd.DataFrame(d)

print(df.std())

print("\n")

print("sixth snipet:")

print("\n")

#product of values

import pandas as pd

import numpy as np

#Create a Dictionary of series

d = {'Name':pd.Series(['sai','nadh','saiman','saiman2k','saiman2kak','akula','sainadh akula',

   'manju','nadh','akula manju','saiman2ka3k','manjunadh akula']),

   'Age':pd.Series([18,20,21,19,20,10,22,18,19,22,21,20]),

   'Rating':pd.Series([9.73,6.84,5.98,8.56,7.20,9.6,5.8,8.78,9.98,6.80,8.10,9.65])

}

#Create a DataFrame

df = pd.DataFrame(d)

print(df.prod())

print("\n")

print("seventh snipet:")

print("\n")

#cummulative sum

import pandas as pd

import numpy as np

#Create a Dictionary of series

d = {'Name':pd.Series(['sai','nadh','saiman','saiman2k','saiman2kak','akula','sainadh akula',

   'manju','nadh','akula manju','saiman2ka3k','manjunadh akula']),

   'Age':pd.Series([18,20,21,19,20,10,22,18,19,22,21,20]),

   'Rating':pd.Series([9.73,6.84,5.98,8.56,7.20,9.6,5.8,8.78,9.98,6.80,8.10,9.65])

}

#Create a DataFrame

df = pd.DataFrame(d)

print(df.cumsum())

print("\n")

print("eight snipet:")

print("\n")

#max value

import pandas as pd

import numpy as np

#Create a Dictionary of series

d = {'Name':pd.Series(['sai','nadh','saiman','saiman2k','saiman2kak','akula','sainadh akula',

   'manju','nadh','akula manju','saiman2ka3k','manjunadh akula']),

   'Age':pd.Series([18,20,21,19,20,10,22,18,19,22,21,20]),

   'Rating':pd.Series([9.73,6.84,5.98,8.56,7.20,9.6,5.8,8.78,9.98,6.80,8.10,9.65])

}

#Create a DataFrame

df = pd.DataFrame(d)

print(df.max())

print("\n")

print("ninth snipet:")

print("\n")

#min value

import pandas as pd

import numpy as np

#Create a Dictionary of series

d = {'Name':pd.Series(['sai','nadh','saiman','saiman2k','saiman2kak','akula','sainadh akula',

   'manju','nadh','akula manju','saiman2ka3k','manjunadh akula']),

   'Age':pd.Series([18,20,21,19,20,10,22,18,19,22,21,20]),

   'Rating':pd.Series([9.73,6.84,5.98,8.56,7.20,9.6,5.8,8.78,9.98,6.80,8.10,9.65])

}

#Create a DataFrame

df = pd.DataFrame(d)

print(df.min())



#1.8-fuction application



import pandas as pd

import numpy as np



def adder(ele1,ele2):

   return ele1+ele2



df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])

df.pipe(adder,2)

print(df.apply(np.mean))

print("\n")

print("second snipet:")

print("\n")

#Row or Column Wise Function Application

import pandas as pd

import numpy as np



df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])

df.apply(np.mean)

print(df.apply(np.mean))

print("\n")

print("third snipet:")

print("\n")

#axis parameter

import pandas as pd

import numpy as np



df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])

df.apply(np.mean,axis=1)

print(df.apply(np.mean))

print("\n")

print("fourth snipet:")

print("\n")

#Element Wise Function Application

import pandas as pd

import numpy as np

df = pd.DataFrame(np.random.randn(5,3),columns=['col1','col2','col3'])

# My custom function

df['col1'].map(lambda x:x*100)

print(df.apply(np.mean))
#1.9-Pandas iteration

print("first snipet:")

#iteritems()

import pandas as pd

import numpy as np

 

df = pd.DataFrame(np.random.randn(4,3),columns=['col1','col2','col3'])

for key,value in df.iteritems():

   print(key,value)

print("\n")

print("second snipet:")

print("\n")

#iterrows()

import pandas as pd

import numpy as np



df = pd.DataFrame(np.random.randn(4,3),columns = ['col1','col2','col3'])

for row_index,row in df.iterrows():

   print(row_index,row)

print("\n")

print("third snipet:")

print("\n")

#itertuples()

import pandas as pd

import numpy as np



df = pd.DataFrame(np.random.randn(4,3),columns = ['col1','col2','col3'])

for row in df.itertuples():

    print(row)
#1.10-Pandas sorting

print("first snipet:")

#By Label

import pandas as pd

import numpy as np



unsorted_df = pd.DataFrame(np.random.randn(10,2),index=[1,4,6,2,3,5,9,8,0,7],columns = ['col2','col1'])



sorted_df=unsorted_df.sort_index()

print(sorted_df)

print("\n")

print("second snipet:")

print("\n")

#order of sorting

import pandas as pd

import numpy as np



unsorted_df = pd.DataFrame(np.random.randn(10,2),index=[1,4,6,2,3,5,9,8,0,7],columns = ['col2','col1'])



sorted_df = unsorted_df.sort_index(ascending=False)

print(sorted_df)

print("\n")

print("third snipet:")

print("\n")

#column sort

import pandas as pd

import numpy as np

 

unsorted_df = pd.DataFrame(np.random.randn(10,2),index=[1,4,6,2,3,5,9,8,0,7],columns = ['col2','col1'])

 

sorted_df=unsorted_df.sort_index(axis=1)



print(sorted_df)

print("\n")

print("fourth snipet:")

print("\n")

#value sorting

import pandas as pd

import numpy as np



unsorted_df = pd.DataFrame({'col1':[2,1,1,1],'col2':[1,3,2,4]})

sorted_df = unsorted_df.sort_values(by='col1')



print(sorted_df)

print("***************")

print("PYTHON PROGRAMS")

print("***************")
#2.1-Prime,Even,Odd

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
#2.2-Calculator



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

#2.3-narcissistic number



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

    print("Given number is narcissistic.")

else:

    print("The number is not narcissistic.")
#2.4-matrices addition-static

x = [[1,2,3],  

     [4,5,6],  

     [7,8,9]]  

  

y = [[9,8,7],  

     [6,5,4],  

     [3,2,1]]  

  

result = [[0,0,0],  

          [0,0,0],  

          [0,0,0]]  

 

for i in range(len(x)):

    for j in range(len(x[0])):  

       result[i][j] = x[i][j] + y[i][j]  

for r in result:  

   print(r)

  
#2.5-array-sort



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
#2.6-Coversions-km,temp



print("ENter the hoices given below:")

print("1.Kilometres to Miles\n2.Celsius to fahrenheit")

a=int(input())

if a==1:

    print("Enter the no of kilometres:")

    b=int(input())

    c=b/0.62137

    print(b," no of kilometres in miles is:",c)

elif a==2:

    print("Enter the temperature{Celsius}:")

    d=float(input())

    e=(d*1.8)+32

    print(d,"in fahrenheit is:",e)
#2.7-Factorial



print("Enter the number to find the factorial:")

a=int(input())

m=1

for i in range(1,a+1):

    m=m*i

print("Factorial of the ",a," is:",m)
#2.8-Constructors

print("first snipet:")

#classes

class area:

    def ar(self):

        print('Enter the length and breadth of the rectangle:')

        l=int(input())

        b=int(input())

        print('Area:',l*b)

a=area()

a.ar()

print("\n")

print("second snipet:")

print("\n")

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

print("\n")

print("Third snipet:")

print("\n")

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

print("\n")

print("fourth snipet:")

print("\n")

#creating more than one constructor in a class

class con:

    def __init__(self):

        print('This is the first constructor')

    def __init__(self):

        print('This is second constructor')

a=con()

#2.9-sets and dictionaries

#SETS

m={"jan","feb","march","aprl"}

print(type(m))

print(m)

#printing using loops

for i in m:

    print(i)

#using add method

m.add("may")

print(m)

#using update function

m.update(["june","july"])

print(m)

#using remove and discard

m.discard("jan")

m.remove("feb")

print(m)

m1={1,2,3,4,5,6}

m2={4,5,6,7,8,9}

#Union,Intersection and Difference

print("Union:")

print(m1|m2)

print('Intersection:')

print(m1&m2)

print('difference between m1 and m2:')

print(m1-m2)

print("\n")

print("**************")

print("dictionaries")

print("**************")

#dictionaries

#dictionaries

d={"Name":"sainadh","Age":"20","Id":"13"}

#creating with dict() keyword

#d1=dict({1:"Akula",2:"SAinadh",3:"13"})

print(d)

#print(d1)

#accessing the values of the dictionary

print("first_Name: %s"% d["Name"])

print("Age: %s"% d["Age"])

print("Id: %s"% d["Id"])

#adding elements to the dictionary

d[1]="25"

print(d)

#deleting elements

del d[1]

print(d)

for i in d:

    print(i)

    print(d[i])
#2.10-tuple

#Tuple

t =(1,2,3)

print(t)

c=0

for i in t:

    print("tuple[%d]= %d"%(c,i))

    c=c+1

#taking elements as input

t1=tuple(input())

print(t1)

c=0

for i in t1:

    print("tuple[%d]=%s"%(c,i))

    c=c+1

#tuple indexing

print(t1[2:4])
#2.11-Generator



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

for i in fact(6):

    print(i)
print("*************")

print("Numpy Library")

print("*************")
#3.1-ndarray,datatype



print("first snipet:")

#creating an 1-d array

import numpy as np 

a = np.array([6013,2018,5060]) 

print(a)

print("\n")

print("second snipet:")

print("\n")

# more than one dimensions 

import numpy as np 

a = np.array([[6013,2018], [5060,13]]) 

print(a)

print("\n")

print("third snipet:")

print("\n")

# minimum dimensions 

import numpy as np 

a = np.array([6013,13,5060,2018], ndmin = 3) 

print(a)

print("\n")

print("fourth snipet:")

print("\n")

#dtype parameter 

import numpy as np 

a = np.array([60.13,50.60,1.3], dtype = complex) 

print(a)

print("\n")

print("fifth snipet:")

print("\n")

#data type

# using array-scalar type 

import numpy as np 

dt = np.dtype(np.int32) 

print(dt)

print("\n")

print("sixth snipet:")

print("\n")

#structured data type 

import numpy as np 

dt = np.dtype([('age',np.int8)]) 

print(dt)
#3.2-array-attributes



print("first snipet:")

# this resizes the ndarray 

import numpy as np 

a = np.array([[6013,13,5060],[5013,6013,2018]]) 

a.shape = (2,3) 

print(a)

print("\n")

print("second snipet:")

print("\n")

#array of evenly spaced numbers 

import numpy as np 

a = np.arange(13) 

print(a)

print("\n")

print("third snipet:")

print("\n")

#ndmin

import numpy as np 

a = np.arange(24) 

a.ndim  

print("\n")

print("fourth snipet:")

print("\n")

# now reshape it 

b = a.reshape(2,4,3) 

print(b)

print("\n")

print("fifth snipet:")

print("\n")

#itemsize

# dtype of array is int8 (1 byte) 

import numpy as np 

x = np.array([1,2,3,4,5], dtype = np.int8) 

print (x.itemsize)

print("\n")

print("sixth snipet:")

print("\n")

#flags

import numpy as np 

x = np.array([1,2,3,4,5]) 

print (x.flags)
#3.3-array from data



print("first snipet:")

# convert list to ndarray 

import numpy as np 

x = [6013,2018,5060] 

a = np.asarray(x) 

print( a)

print("\n")

print("second snipet:")

print("\n")

# dtype is set 

import numpy as np 



x = [6013,2018,5060]

a = np.asarray(x, dtype = float) 

print (a)

print("\n")

print("third snipet:")

print("\n")



# ndarray from tuple 

import numpy as np 



x = (6013,5060,2018) 

a = np.asarray(x) 

print(a)

print("\n")

print("fourth snipet:")

print("\n")



# ndarray from list of tuples 

import numpy as np 



x = [(13,6013,2018),(5060,6013)] 

a = np.asarray(x) 

print(a)

print("\n")

print("fifth snipet:")

print("\n")



# obtain iterator object from list 

import numpy as np 

list = range(5) 

it = iter(list)  

# use iterator to create ndarray 

x = np.fromiter(it, dtype = float) 

print(x)
#3.4-indexing and slicing-advanced



print("first snipet:")

#indexing

import numpy as np 

a = np.arange(10) 

s = slice(2,8,1) 

print(a[s])

print("\n")

print("second snipet:")

print("\n")

#start,stop,step

import numpy as np 

a = np.arange(10) 

b = a[2:9:1] 

print(b)

print("\n")

print("third snipet:")

print("\n")

# slice items starting from index 

import numpy as np 

a = np.arange(10) 

print (a[2:])

print("\n")

print("fourth snipet:")

print("\n")

# slice items between indexes 

import numpy as np 

a = np.arange(13) 

print (a[6:13])

print("\n")

print("fifth snipet:")

print("\n")

#slicing to an array

import numpy as np 

a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 

print (a)  

# slice items starting from index

print (a[1:])

print("\n")

print("sixth snipet:")

print("\n")

import numpy as np 

x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 

   

print ('Our array is:')

print (x) 

print ('\n') 



rows = np.array([[0,0],[3,3]])

cols = np.array([[0,2],[0,2]]) 

y = x[rows,cols] 

   

print ('The corner elements of this array are:') 

print (y)

print("\n")

print("seventh snipet:")

print("\n")

#boolean array indexing

import numpy as np 

x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 



print ('Our array is:') 

print (x)

print ('\n')  



# Now we will print the items greater than 5 

print ('The items greater than 5 are:') 

print (x[x > 5])
#3.5-broadcasting

#broadcasting

import numpy as np 

a = np.array([60,13,20,18]) 

b = np.array([10,20,30,40]) 

c = a * b 

print(c)

print("\n")

print("second snipet:")

print("\n")

import numpy as np 

a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 

b = np.array([1.0,2.0,3.0])  

   

print ('First array:') 

print (a )

print ('\n') 

   

print ('Second array:') 

print (b) 

print ('\n')  

   

print ('First Array + Second Array') 

print (a + b)
#3.6-iterating over array



print("first snipet:")

#ordering

import numpy as np 

a = np.arange(0,60,5) 

a = a.reshape(3,4) 

   

print ('Original array is:')

print (a) 

print ('\n')  

   

print ('Transpose of the original array is:') 

b = a.T 

print (b) 

print ('\n')

   

print ('Modified array is:') 

for x in np.nditer(b): 

   print (x,)

print("\n")

print("second snipet:")

print("\n")

#nditer

import numpy as np 

a = np.arange(0,60,5) 

a = a.reshape(3,4) 



print ('Original array is:') 

print (a) 

print ('\n')  



print ('Sorted in C-style order:') 

for x in np.nditer(a, order = 'C'): 

   print (x,)  

print ('\n') 



print ('Sorted in F-style order:') 

for x in np.nditer(a, order = 'F'): 

   print (x,)

print("\n")

print("third snipet:")

print("\n")

#broadcasting iteration

import numpy as np 

a = np.arange(0,60,5) 

a = a.reshape(3,4) 



print ('First array is:') 

print (a) 

print ('\n')  



print ('Second array is:') 

b = np.array([1, 2, 3, 4], dtype = int) 

print (b)  

print ('\n') 



print ('Modified array is:') 

for x,y in np.nditer([a,b]): 

   print ("%d:%d" % (x,y),)
#3.7-string functions

print("first snipet")

#string concetanation

import numpy as np 

print ('Concatenate two strings:') 

print (np.char.add(['Sainadh'],['Akula'])) 

print ('\n')

print ('Concatenation example:') 

print (np.char.add(['sai', 'nadh'],['akula', ' saiman']))

print("\n")

print("second snipet:")

print("\n")

#string multiply

import numpy as np 

print (np.char.multiply('Sainadh ',3))

print("\n")

print("third snipet:")

print("\n")

#string centre

import numpy as np 

# np.char.center(arr, width,fillchar) 

print (np.char.center('saiman2k', 30,fillchar = '$'))

print("\n")

print("fourth snipet:")

print("\n")

#capitalise

import numpy as np 

print (np.char.capitalize('akulasainadh'))

print("\n")

print("fifth snipet:")

print("\n")

#string title

import numpy as np 

print (np.char.title('myself sainadh akula'))

print("\n")

print("sixth snipet:")

print("\n")

#string split

import numpy as np 

print (np.char.split ('my name is sainadh')) 

print (np.char.split ('sainadh,akula,saiman2k', sep = ','))

print("\n")

print("seventh snipet:")

print("\n")

#string strip

import numpy as np 

print (np.char.strip('sainadh','akula')) 

print (np.char.strip(['saiman','saiman2k','akula'],'a'))

print("\n")

print("eight snipet:")

print("\n")

#string replace

import numpy as np 

print (np.char.replace ('Myself sainadh akula', 'akula', 'saiman2k'))

#3.8-mathematical functions

print("first snipet:")

#trignometric values

import numpy as np 

a = np.array([0,30,45,60,90]) 



print ('Sine of different angles:') 

# Convert to radians by multiplying with pi/180 

print (np.sin(a*np.pi/180)) 

print ('\n')  



print ('Cosine values for angles in array:') 

print (np.cos(a*np.pi/180)) 

print ('\n')  



print ('Tangent values for given angles:') 

print (np.tan(a*np.pi/180))

print("\n")

print("second snipet:")

print("\n")

#rounding

import numpy as np 

a = np.array([1.0,5.55, 123, 0.567, 25.532]) 



print ('Original array:') 

print (a) 

print ('\n')  



print ('After rounding:') 

print (np.around(a)) 

print (np.around(a, decimals = 1)) 

print (np.around(a, decimals = -1))

print("\n")

print("third snipet:")

print("\n")

#floor

import numpy as np 

a = np.array([-1.7, 1.5, -0.2, 0.6, 10]) 



print ('The given array:') 

print (a) 

print ('\n')  



print ('The modified array:') 

print (np.floor(a))

print("\n")

print("fourth snipet:")

print("\n")

#ceil

import numpy as np 

a = np.array([-1.7, 1.5, -0.2, 0.6, 10]) 



print ('The given array:') 

print (a) 

print ('\n')  



print ('The modified array:') 

print (np.ceil(a))
#3.9-arithmatic operations

print("first snipet:")

#arithmatic operations

import numpy as np 

a = np.arange(9, dtype = np.float_).reshape(3,3) 



print ('First array:') 

print (a) 

print ('\n')  



print ('Second array:') 

b = np.array([10,10,10]) 

print (b) 

print ('\n')  



print ('Add the two arrays:') 

print (np.add(a,b)) 

print ('\n')  



print ('Subtract the two arrays:') 

print (np.subtract(a,b)) 

print ('\n')  



print ('Multiply the two arrays:') 

print (np.multiply(a,b)) 

print ('\n' ) 



print ('Divide the two arrays:') 

print (np.divide(a,b))

print("\n")

print("second snipet:")

print("\n")

#reciprocal

import numpy as np 

a = np.array([0.25, 1.33, 1, 5, 100]) 



print ('Our array is:') 

print (a) 

print ('\n')  



print ('After applying reciprocal function:') 

print (np.reciprocal(a)) 

print ('\n')  



b = np.array([100], dtype = int) 

print ('The second array is:') 

print (b) 

print ('\n')  



print ('After applying reciprocal function:') 

print (np.reciprocal(b))

print("\n")

print("third snipet:")

print("\n")

#power

import numpy as np 

a = np.array([10,100,1000]) 



print ('Our array is:') 

print (a) 

print ('\n')  



print ('Applying power function:') 

print (np.power(a,2)) 

print ('\n')  



print ('Second array:') 

b = np.array([1,2,3]) 

print (b) 

print ('\n')  



print ('Applying power function again:') 

print (np.power(a,b))

print("\n")

print("fourth snipet:")

print("\n")

#mod

import numpy as np 

a = np.array([10,20,30]) 

b = np.array([3,5,7]) 



print ('First array:') 

print (a) 

print ('\n')  



print ('Second array:') 

print (b )

print ('\n')  



print ('Applying mod() function:' )

print (np.mod(a,b)) 

print ('\n')  



print ('Applying remainder() function:') 

print (np.remainder(a,b))
#3.10-statistical functions

print("first snipet:")

#amin() and amax()

import numpy as np 

a = np.array([[1,2,3],[4,5,6],[7,8,9]]) 



print ('Our array is:') 

print (a)  

print ('\n')  



print ('Applying amin() function:') 

print (np.amin(a,1)) 

print ('\n')  



print ('Applying amin() function again:') 

print (np.amin(a,0)) 

print ('\n')  



print ('Applying amax() function:') 

print (np.amax(a)) 

print ('\n' ) 



print ('Applying amax() function again:') 

print (np.amax(a, axis = 0))

print("\n")

print("second snipet:")

print("\n")

#ptp()

import numpy as np 

a = np.array([[1,2,3],[4,5,6],[7,8,9]]) 



print ('Our array is:' )

print (a) 

print ('\n')  



print ('Applying ptp() function:') 

print (np.ptp(a)) 

print ('\n')  



print ('Applying ptp() function along axis 1:') 

print (np.ptp(a, axis = 1) )

print ('\n' )  



print ('Applying ptp() function along axis 0:')

print (np.ptp(a, axis = 0) )

print("\n")

print("third snipet:")

print("\n")

#percentile

import numpy as np 

a = np.array([[1,2,3],[4,5,6],[7,8,9]]) 



print ('Our array is:') 

print (a) 

print ('\n')  



print ('Applying percentile() function:') 

print (np.percentile(a,50)) 

print ('\n')  



print ('Applying percentile() function along axis 1:') 

print (np.percentile(a,50, axis = 1)) 

print ('\n')  



print ('Applying percentile() function along axis 0:') 

print (np.percentile(a,50, axis = 0))

print("\n")

print("fourth snipet:")

print("\n")

#median

import numpy as np 

a = np.array([[1,2,3],[4,5,6],[7,8,9]])



print ('Our array is:') 

print (a) 

print ('\n')  



print ('Applying median() function:') 

print (np.median(a) )

print ('\n')  



print ('Applying median() function along axis 0:') 

print (np.median(a, axis = 0)) 

print ('\n')  

 

print ('Applying median() function along axis 1:') 

print (np.median(a, axis = 1))

print("\n")

print("fifth snipet:")

print("\n")

#mean

import numpy as np 

a = np.array([[1,2,3],[4,5,6],[7,8,9]]) 



print ('Our array is:') 

print (a) 

print ('\n')  



print ('Applying mean() function:') 

print (np.mean(a) )

print ('\n')  



print ('Applying mean() function along axis 0:') 

print (np.mean(a, axis = 0)) 

print ('\n')  



print ('Applying mean() function along axis 1:') 

print (np.mean(a, axis = 1))