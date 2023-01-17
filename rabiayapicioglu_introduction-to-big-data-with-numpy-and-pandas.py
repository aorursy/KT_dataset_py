# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from IPython.display import Image

import os

!ls ../input/



Image("../input/images/bda-696x394.jpg")
from IPython.display import Image

import os

!ls ../input/



Image("../input/images/history-bigdata.jpg")
from IPython.display import Image

import os

!ls ../input/



Image("../input/images/threev.png")
from IPython.display import Image

import os

!ls ../input/



Image("../input/images/Management.png")
from IPython.display import Image

import os

!ls ../input/



Image("../input/images/company.jpg")
from IPython.display import Image

import os

!ls ../input/



Image("../input/images/Trump_New_York_Times_tweet_.jpg")
import numpy as np #importing the Numpy library



array = np.array([1,2,3]) #creating the one dimension numpy array -1 row and 3 colums

print( array ) #let's see what we've created by using print command

# numpy arrays with more than one dimensions 

import numpy as np 

array = np.array([[1, 2], [3, 4]]) #creating the numpy array which has 2 rows and 2 colums 

print( array )
# dtype parameter 

import numpy as np 

array = np.array([1, 2, 3], dtype = complex) #dtype can take different parameters according to your array members

print( array )
import numpy as np 

array = np.array([[1,2,3],[4,5,6]]) #we can see how many rows and colums are there by using shape method

print(array.shape)
#we want to resize the array,here ew want to make it with 3 rows and 2 colums

import numpy as np 



a = np.array([[1,2,3],[4,5,6]])

print( "Array before resizing:\n ", a )

a.shape = (3,2) 

print("Array after resizing:\n ", a ) 



#Also we can use the reshape methode for resizing



#import numpy as np 

#a = np.array([[1,2,3],[4,5,6]]) 

#b = a.reshape(3,2) 

#print( b )
# this is one dimensional array-members starting from 1 and ends at 23 so don't forget 24 is exclusive 

import numpy as np 

array= np.arange(24) 

print( array ) # to see result we can just use the name of the array



#print( type( array )) we can see the types by using type() method



# dtype of array is now float32 (4 bytes)  

#x = np.array([1,2,3,4,5], dtype = np.float32) 

#print( x.itemsize ) #if we want to learn the size of item 

p1=[] # we have created an empty list

p1.append(4) #append int 4 at the end of the array

p1.append(5) #append int 5 at the end of the array 

print( p1 )
# now reshape it 

b = array.reshape(2,4,3) 

print( b ) 

# b is having three dimensions
#we can create empty arrays in numpy

#numpy.empty(shape, dtype = float, order = 'C')

import numpy as np 

x = np.empty([3,2], dtype = int) 

print( x )
#Also we can create array of just zeros or ones

#numpy.zeros(shape, dtype = float, order = 'C')

#numpy.ones(shape, dtype = None, order = 'C')



# array of five zeros. Default dtype is float 

import numpy as np 

x = np.zeros(5) 

print( x )
#numpy.linspace(start, stop, num, endpoint, retstep, dtype)

import numpy as np 

x = np.linspace(10,20,5) 

print( x )
# endpoint set to false 

#endpoint

#True by default, hence the stop value is included in the sequence. If false, it is not included

import numpy as np 

x = np.linspace(10,20, 5, endpoint = False) 

print( x )
#slicing arrays start point-end point-difference

import numpy as np 

a = np.arange(10) 



print( a[0:3]) #starting from 0 and ends at 2 ,here 3 is exclusive



print( a[:3] ) #starts once again from the zero if we had not written anything before : 



print( a[:-1]) #starts from 0 and ends at last member from the end point ,8.



reverse_array=a[::-1] # we can reverse array in this way

print( reverse_array )
import numpy as np 



array=np.array([[1,2,3,4,5],[6,7,8,9,10]]) # we have created an array with two rows and two columns



print( array[1,1]) # we can acces members in this way

print("------------")

print( array[:,[1,2]]) #we take all of the rows  and first and second of the columns in this way

print("------------")

print( array[-1,:] ) #we take last row and all of the columns

print("------------")

print( array[:,-1] )#we take all of the rows and the last column
import numpy as np 

#flatten format and transposing of arrays

array=np.array([[1,2,3],[4,5,6],[7,8,9]]) # we have created an array with three rows and three columns



flatten=array.ravel() #if we want to get rid of all of these dimension structure use ravel() method

print( flatten )

print("-------------")



transposed=array.T #we can alsso get the transpose of the given array in this way

print( transposed )
#stacking arrays



array1=np.array( [[1,2],[3,4]])

array2=np.array( [[-1,-2],[-3,-4]])



#vertical stacking



Vstacked=np.vstack((array1,array2))

print( Vstacked )

print("------------")

#horizontal stacking



Hstacked=np.hstack((array1,array2))

print( Hstacked )
#copy and convert arrays



list1=[1,2,3,4]



array=np.array(list1) # we can also create a numpy array by using the already created list



#what happens if we run the following 



referenced1=array

referenced2=referenced1



referenced2[0]=7



print( referenced1 )

print( array ) # we see that they are referenced of each other if we assigned them to each other directly

#how to get rid of from that situation



copy1=array.copy()

copy1[0]=9

print( array )# this time array has not been changed
#numpy operations



a=np.array([1,2,3])

b=np.array([4,5,6])



#we can make mathematical operations on arrays in the following ways

print(a+b) #summation

print(a-b) #difference

print(a*b) #mutiplication

print(a**2+1) #exponential operations

print(np.sin(a)) #trigonometric operations

print(a<2) #boolean operations individually

import numpy as np

#element wise product with arrays

a=np.array([[1,2,3],[4,5,6]])

b=np.array([[7,8,9],[5,7,8]])



print( a*b )

#--------------------------------

#matrix multiplication

print("-----------------")

#a.dot(b) gets value error,cause we cannot multiply two arrays with dimensions (2x3) and (2x3)

a.dot(b.T) #works well case we are multipliying two arrays with dimensions (2x3) and (3x2 ) which is transposed

print(a)

print("-----------------")

print(np.exp(a))

#we can use other methods like max,min and sum

print(a.sum(),'\n')

print(a.max(),'\n')

print(a.min(),'\n')



import numpy as np

#element wise product with arrays

a=np.array([[1,2,3],[4,5,6]])



print( a.sum( axis=0 ))

print( a.sum( axis=1 ))

print( np.sqrt(a))

#or instead of a+a we can use



np.add(a,a)
#pandas.Series( data, index, dtype, copy)

#A series can be created using various inputs like −

   #Array

   #Dict

   #Scalar value or constant

#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

data = np.array(['a','b','c','d'])

s = pd.Series(data)

print( s )
#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

data = np.array(['a','b','c','d'])

s = pd.Series(data,index=[100,101,102,103]) # we can also use set_index() method later on

print(s)
#create series from dictionaries

#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

data = {'a' : 0., 'b' : 1., 'c' : 2.} #this is a normal python dictionary

s = pd.Series(data) #we've used dictionary to create pandas series

print( s )
#Create a Series from Scalar



#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

s = pd.Series(5, index=[0, 1, 2, 3])

print( s )
#pandas.DataFrame( data, index, columns, dtype, copy)

#A pandas DataFrame can be created using various inputs like −

  #Lists

  #dict

  #Series

  #Numpy ndarrays

  #Another DataFrame

import pandas as pd

data = [['Alex',10],['Bob',12],['Clarke',13]]

print( type( data ))#we've used list to create data frame for example

df = pd.DataFrame(data,columns=['Name','Age'])

print( df )
#Create a DataFrame from Dict of ndarrays / Lists



import pandas as pd

data = {'Name':['Tomy', 'John', 'Stevens', 'Rownie'],'Age':[28,34,29,42]}

df = pd.DataFrame(data)

print( df )
#Create a DataFrame from List of Dicts



import pandas as pd

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

df = pd.DataFrame(data)

print( df )
#Column Selection

import pandas as pd



d = {'first' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),

   'second' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}



df = pd.DataFrame(d)

print( df['first'] )

#or we do

df #we can see it in the table format
#Column Addition

import pandas as pd



d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),

   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}



df = pd.DataFrame(d)



# Adding a new column to an existing DataFrame object with column label by passing new series



print ("Adding a new column by passing as Series:")

df['three']=pd.Series([10,20,30],index=['a','b','c'])

print( df )



print ("Adding a new column using the existing columns in DataFrame:")

df['four']=df['one']+df['three']



print( df )

df
# column Deletion

# Using the previous DataFrame, we will delete a column

# using del function

import pandas as pd



d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 

   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']), 

   'three' : pd.Series([10,20,30], index=['a','b','c'])}



df = pd.DataFrame(d)

print ("Our dataframe is:")

print( df )



# using del function

print ("Deleting the first column using DEL function:")

del df['one']

print( df )



# using pop function

print ("Deleting another column using POP function:")

df.pop('two')

print( df )
#Row Selection, Addition, and Deletion

import pandas as pd



d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 

     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}



df = pd.DataFrame(d)

print( df.loc['b'] )
#Slice Rows



import pandas as pd



d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 

   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}



df = pd.DataFrame(d)

print( df )

print( df.loc[:,["one"]] ) #all rows and column one



print( df.loc[::-1,:])#reversed rows and all columns



print( df.iloc[:,1])#prints indexes instead of names

#Addition of Rows

#Add new rows to a DataFrame using the append function. This function will append the rows at the end.

import pandas as pd



df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])

df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])



df = df.append(df2)

print( df )
#Deletion of Rows



import pandas as pd



df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])

df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])



df = df.append(df2)



# Drop rows with label 0

df = df.drop(0)



print( df )
#Filters

import pandas as pd



df = pd.DataFrame([[1, 2,3,4], [3, 4,5,6], [6, 7,8,9]], columns = ['a','b','c','d'])

print(df)



filter1=df.a>2



print( df[ filter1 ]) #usage of filters



filter2=df.b<=5



print( df[ filter1 & filter2 ]) #combining two filters

print( df[ df.a > 5]) #we can use it briefly like

#list comprehension

import pandas as pd



df = pd.DataFrame([[1, 2,3,4], [3, 4,5,6], [6, 7,8,9]], columns = ['a','b','c','d'])

print(df)



avg_a=df.a.mean() #we've found the mean of the column a

print( avg_a)



df["avg"]=["less" if avg_a > each else "much" for each in df.a] #we are adding a new column called avg

df

import pandas as pd



df = pd.DataFrame([[1, 2,3,4], [3, 4,5,6], [6, 7,8,9]], columns = ['stock value','stock index','stock place no','zip code'])

print(df)

#changing the column names which cannot be used easily



df.columns=[ each.split()[0]+"_"+each.split()[1] if len( each.split()) > 1 else each for each in df.columns]

print(df)#we have successfully changed the column names

df.drop(["zip_code"],axis=1,inplace=True)#this is a permanent change in the table ,use inplace=True

print(df) #zip code has gone then
import pandas as pd



df = pd.DataFrame([[1, 2,3,4], [3, 4,5,6], [6, 7,8,9],[64, 75,87,98],[9,8,7,5],[63, 72,81,91]], columns = ['stock value','stock index','stock place no','zip code'])

print(df)



first5=df.head() #prints the first five values

last5=df.tail() #prints the last five values

data_concat1=pd.concat([first5,last5],axis=0) #combine rows together if axis=0

data_concat2=pd.concat([first5,last5],axis=1) #combie columns together if axis=1



print( first5 )

print( last5  )

print(data_concat1)

print( data_concat2)

#transforming data



df['stock value']=[each*2 for each in df['stock value']] #multiply all elements with 2

df

#or transform with apply method



def multiplyby2( item ):

    return item*2

df["stock value"]=df["stock value"].apply( multiplyby2 )

df
df.describe()