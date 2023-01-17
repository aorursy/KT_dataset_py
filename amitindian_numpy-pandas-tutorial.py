import sys

import time

import numpy as np
Size = 10000                # Python List

time_python = time.time()

L1 = range(Size)

L2 = range(Size)

result_python =[L1[i] + L2[i] for i in range(len(L1))]

print((time.time()-time_python)*1000)
time_numpy= time.time()      # Numpy Array

A1 = np.arange(Size)

A2 = np.arange(Size)

result_numpy = A1+A2

print((time.time()-time_numpy)*1000)
import numpy as np
my_list =[1,2,3,4]                      # This is an example of python list

my_list
type(my_list)                           # Type function is used to know the type of Python objects
array =np.array(my_list)                # This is a one dimensional array

array
multiple_list =[[1,2,3],[4,5,6],[7,8,9],[10,11,12]] # Creating a 2D list or list of lists

multiple_list
matrix = np.array(multiple_list)

matrix
array.shape            # This gives the shape of the array
matrix.shape           # This gives the number of rows and columns of the array.
array.ndim            # This is a 1 dimensional array
matrix.ndim          # This is a 2 dimensional array or matrix
array.dtype
matrix.dtype
np.arange(0,15)
np.arange(0,10,2)
np.zeros(5)
np.ones(5)
np.ones((2,4))
np.zeros((3,3))
np.linspace(0,3,9)
np.eye(4)
np.random.rand(40)
np.random.rand(2,2)
np.random.rand(4,4)
np.random.rand(9).reshape(3,3)
my_array =np.arange(10)                     # It will return an array from 0 to 10

my_array
my_array[4]                                  # It will return element at index 10
my_array[1:8]                       # It will return all the elements between 1 and 8 excluding 8
my_array[8:]                        # It will return all elements from index 8 and beyond
my_array[:6]                         # It will return all elements from first index to 5
my_array[0:6] = 6

my_array
array_condi = np.arange(0,11)

array_condi
array_condi>5                      # This will return a boolean array
boolian = array_condi>5           # This will return all the elements of an array where element size > 5

array_condi[boolian]
array_condi[array_condi>5]        # This is the same thing without using another object.
array_2D = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

array_2D
array_2D.shape
array_2D.ndim
scaler = 2

array_1D = np.array([12,14,16])

array_XD = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
array_1D.shape
array_2D/scaler            # operation with a scaler
array_2D - array_1D           # operation with a array of different shape
array = np.array([[1,1,1,1]])

print(array,array_2D)
array = np.arange(1,11)

array
array.min()
array.max()
array.argmin()       # Index position of minimum of array
array.argmax()         # Index position of maximum of array
np.sqrt(array)                   # To calculate square root of all elements in an array
array.mean()                         # To calculate mean of all the values in an array
np.exp(array)                         # To calculate exponential value of each element in an array
array = np.arange(0,16)                     # Using reshape we can change the dimensions of the array

array_2d = array.reshape(4,4)

array_2D
array_2D.flatten()                         # Flatten is used to convert a 2D array to 1D array
array_2D.transpose()                     # Transpose is used to convert the rows into columns and vice-versa
array_x = np.array([[1,2,3,4],[5,6,7,8]])                   # Lets create 2 arrays

array_y = np.array([[10,11,12,13],[14,15,16,17]])                                         
np.concatenate((array_x,array_y), axis =1)                  # Join 2 arrays along columns
array_z = np.concatenate((array_x,array_y), axis =0) 

array_z
np.hsplit(array_z,2)                # It will split the array into 2 equal halves along the columns
np.vsplit(array_z,2)               # It will split the array into 2 equal halves along the rows
import pandas as pd

import numpy as np
xyz = {'Day':[1,2,3,4,5,6,7,8,9,10],

       "Visitors":[100,200,300,400,500,600,700,800,900,1000],

       'Bouncerate':[15,20,25,15,30,25,35,65,95,100]}
df = pd.DataFrame(xyz)

df
df.head(5)
df.tail(5)
df1= pd.DataFrame({"Emp":[1,2,3,4],

                   "Salary":[60000,70000,80000,90000],

                   "Allowance":[5000,4500,4500,6700],

                  "Name":['A','B','C','D']

                  })

df2= pd.DataFrame({"Emp":[1,2,3,7],

                   "Salary":[60000,70000,80000,90000],

                   "Allowance":[5000,4500,4500,6700],

                  'Name':['A','F','E','D']

                  })
pd.merge(left =df2 , right =df1, on ='Emp')
pd.merge(right =df2 , left =df1, on ='Name')
pd.concat([df1,df2],axis =0)              # Joining two DataFrame along the rows
pd.concat([df1,df2],axis =1)    # Joining 2 DataFrame along the columns
df1[:3]                    # Accessing the first 2 elements in series
df1[3:4]                      # Accessing all the elements from 3rd to 4th index
df1['Emp']                    # Accessing the data using labels
df1+df2                       # Performing element wise mathematical operations on series
df1.shape                    # shape function is used to know the dimensions
df1.dtypes                    # dtypes function for information about layout 
df1.info()
df1.count()
df1.index
df1.columns
New_df1 = df1.set_index('Emp')

New_df1
df1['Salary']                      # Column Selection
df1['Incentives'] =[1000,2000,2500,3000]            # Adding a new column

df1.set_index('Emp')
df1['Pay_Sum']= df1['Salary']+ df1['Incentives'] # Addition of two columns. You can perform any math operation. 
New_df1 =df1

New_df1
New_df1.drop('Pay_Sum',axis =1,inplace =True)          # Drop Sum and modify the dataframe

New_df1
del New_df1['Incentives']                 # Column Deletion using del

New_df1
New_df1.drop('Salary',1)                     # Dropping 'Salary' column

New_df1.set_index('Name')
sm_df = pd.read_csv('Social_Network_Ads.csv')

sm_df.set_index('Gender').head()
sm_df.iloc[0:5,0:2]                # It will return rows from 0-5 and columns from 0-2
sm_df.iloc[:,0:4]                    # Return the columns from 0-4
sm_df.iloc[0:6,:]                 # Return all the rows from 0-6
sm_df.loc[0:5,'Age']            # Returns Age 1-5 and column Address
sm_df.loc[0:6,'Gender':'Purchased']          # Returns Index 1-6 and columns Address to Country
sm_df.loc[:,"Age":]          # Return all the columns from Age onwards and all the rows
sm_df['Age']>15                    # This returns a Boolean Series
sm_df[sm_df['Age']>=20]          # This returns all the rows for which the condition is True
sm_df[(sm_df['Age']>=15) & (sm_df['Gender']!='Male')]   # It will return rows where both the conditions are satisfied.
group = sm_df.iloc[:,1:].set_index('Gender').head()

group
by_Age=group.groupby('Gender')
by_Age.mean()           # Average bill
by_Age.sum()                     # Sum
sm_df.head()
sm_df["Age"].unique()                # Observe the unique values of tips['Age']
sm_df['Gender'].value_counts()       # Observe the number of unique values of tips['Gender']
sm_df['Age'].nunique()           # Observe the number of unique values of tips['Age']
df1
def times2(x):                      # apply this function

    return x*2
df1['Salary'].apply(times2)       # Applying times2() function on a column of dataframe
df1['Salary'].apply(lambda x : x * 2)    # Applying lambda function on a column of dataframe