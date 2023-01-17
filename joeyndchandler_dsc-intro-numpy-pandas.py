#load the library and check its version, just to make sure we aren't using an older version

import numpy as np #np is an alias for numpy

np.__version__

'1.12.1'
#create a list comprising numbers from 0 to 9

L = list(range(10))



L #this line will automatically print we don't have to use the print function in the last line
#converting integers to string - this style of handling lists is known as list comprehension.

#List comprehension offers a versatile way to handle list manipulations tasks easily. We'll learn about them in future tutorials. Here's an example.  

[str(c) for c in L]
#All items in L are of integeer type

[type(item) for item in L]
#creating arrays

np.zeros(10, dtype='int')
#creating a 3 row x 5 column matrix

np.ones((3,5), dtype=float)
#creating a matrix with a predefined value

np.full((3,5),1.23)
#create an array with a set sequence

np.arange(0, 20, 2)
#create an array of even space between the given range of values

np.linspace(0, 1, 11)
#create a 3x3 array with mean 0 and standard deviation 1 in a given dimension

np.random.normal(0, 1, (3,3))
#create an identity matrix

np.eye(3)
#set a random seed

np.random.seed(1)
x1 = np.random.randint(20, size=6) #one dimension

x2 = np.random.randint(20, size=(3,4)) #two dimension



print("x1 ndim:", x1.ndim)

print("x1 shape:", x1.shape)

print("x1 size: ", x1.size)

print(x1)

print(" ")

print("x2 ndim:", x2.ndim)

print("x2 shape:", x2.shape)

print("x2 size: ", x2.size)

print(x2)



#All values will be between 0-19 inclusive
x1 = np.array([4, 3, 4, 4, 8, 4])

x1
#assess value to index zero

x1[0]
#assess fifth value

x1[4]
#get the last value

x1[-1] #negative indexing
#get the second last value

x1[-2]
#in a multidimensional array, we need to specify row and column index

x2
#3rd row and 4th column value

x2[2,3]
#3rd row and last value from the 3rd column

x2[2,-1]
#replace value at 0,0 index

x2[0,0] = 12

x2
x = np.arange(10)

x
#from start to 4th position

x[:5]
#from 4th position to end

x[4:]
#from 4th to 6th position

x[4:7]
#return elements at even place

x[ : : 2]
#return elements from first position step by two

x[1::2]
#reverse the array

x[::-1]
#You can concatenate two or more arrays at once.

x = np.array([1, 2, 3])

y = np.array([3, 2, 1])

z = [21,21,21]

np.concatenate([x, y,z])
#You can also use this function to create 2-dimensional arrays.

grid = np.array([[1,2,3],[4,5,6]])

np.concatenate([grid,grid])
#Using its axis parameter, you can define row-wise or column-wise matrix

np.concatenate([grid,grid],axis=1)
x = np.array([3,4,5])

grid = np.array([[1,2,3],[17,18,19]])

np.vstack([x,grid])
#Similarly, you can add an array using np.hstack

z = np.array([[9],[9]])

np.hstack([grid,z])
x = np.arange(10)

x
x1,x2,x3,x4 = np.split(x,[2,4,6])

print(x1,x2,x3,x4)
grid = np.arange(16).reshape((2,8))

grid
grid = np.arange(16).reshape((4,4))

grid
upper,lower = np.vsplit(grid,[3])

print (upper)

print(" ")

print (lower)
#load library - pd is just an alias. I used pd because it's short and literally abbreviates pandas.

#You can use any name as an alias. 

import pandas as pd
#create a data frame - dictionary is used here where keys get converted to column names and values to row values.

data = pd.DataFrame({'Country': ['Russia','Colombia','Chile','Equador','Nigeria'],

                    'Rank':[121,40,100,130,11]})

data
#We can do a quick analysis of any data set using:

#By default only numeric fields are returned.

data.describe()
#If we want non-numeric fields as well

data.describe(include='all')
#Among other things, it shows the data set has 5 rows and 2 columns with their respective names.

data.info()
#Let's create another data frame.

data = pd.DataFrame({'group':['a', 'a', 'a', 'b','b', 'b', 'c', 'c','c'],'ounces':[4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

data
#Let's sort the data frame by ounces - inplace = True will make changes to the data

data.sort_values(by=['ounces'],ascending=True,inplace=False)
data.sort_values(by=['group','ounces'],ascending=[True,False],inplace=False)

#create another data with duplicated rows

data = pd.DataFrame({'k1':['one']*3 + ['two']*4, 'k2':[4,4,3,3,3,2,1]})

data
#sort values 

data.sort_values(by='k2')
#remove duplicates - ta da! 

data.drop_duplicates()
data.drop_duplicates(subset='k1')
data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami','corned beef', 'Bacon', 'pastrami', 'honey ham','nova lox'],

                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

data
meat_to_animal = {

'bacon': 'pig',

'pulled pork': 'pig',

'pastrami': 'cow',

'corned beef': 'cow',

'honey ham': 'pig',

'nova lox': 'salmon'

}



def meat_2_animal(series):

    if series['food'] == 'bacon':

        return 'pig'

    elif series['food'] == 'pulled pork':

        return 'pig'

    elif series['food'] == 'pastrami':

        return 'cow'

    elif series['food'] == 'corned beef':

        return 'cow'

    elif series['food'] == 'honey ham':

        return 'pig'

    else:

        return 'salmon'





#create a new variable

data['animal'] = data['food'].map(str.lower).map(meat_to_animal)

data
#another way of doing it is: convert the food values to the lower case and apply the function

lower = lambda x: x.lower()

data['food'] = data['food'].apply(lower)

data['animal2'] = data.apply(meat_2_animal, axis='columns')

data
data.assign(new_variable = data['ounces']*10)

#Let's remove the column animal2 from our data frame.

data.drop('animal2',axis='columns',inplace=True)

data
#Series function from pandas are used to create arrays

data = pd.Series([1., -999., 2., -999., -1000., 3.])

data
#replace -999 with NaN values

data.replace(-999, np.nan,inplace=True)

data
#We can also replace multiple values at once.

data = pd.Series([1., -999., 2., -999., -1000., 3.])

data.replace([-999,-1000],np.nan,inplace=True)

data
#Now, let's learn how to rename column names and axis (row names).



data = pd.DataFrame(np.arange(12).reshape((3, 4)),index=['Ohio', 'Colorado', 'New York'],columns=['one', 'two', 'three', 'four'])

data
#Using rename function

data.rename(index = {'Ohio':'SanF'}, columns={'one':'one_p','two':'two_p'},inplace=True)

data
#You can also use string functions

data.rename(index = str.upper, columns=str.title,inplace=True)

data
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],

                   'key2' : ['one', 'two', 'one', 'two', 'one'],

                   'data1' : np.random.randn(5),

                   'data2' : np.random.randn(5)})

df
#calculate the mean of data1 column by key1

grouped = df['data1'].groupby(df['key1'])

grouped.mean()
dates = pd.date_range('20130101',periods=6)

df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))

df
#get first n rows from the data frame

df[:3]
#slice based on date range

df['20130101':'20130104']
#slicing based on column names

df.loc[:,['A','B']]
#slicing based on both row index labels and column names

df.loc['20130102':'20130103',['A','B']]
#slicing based on index of columns

df.iloc[3] #returns 4th row (index is 3rd)
#returns a specific range of rows

df.iloc[2:4, 0:2]
#returns specific rows and columns using lists containing columns or row indexes

df.iloc[[1,5],[0,2]] 
df[df.A > 1]

#we can copy the data set

df2 = df.copy()

df2['E']=['one', 'one','two','three','four','three']

df2
#select rows based on column values

df2[df2['E'].isin(['two','four'])]
#select all rows except those with two and four

df2[~df2['E'].isin(['two','four'])]
#list all columns where A is greater than C

df.query('A > C')
#using OR condition

df.query('A < B | C > A')
#create a data frame

data = pd.DataFrame({'group': ['a', 'a', 'a', 'b','b', 'b', 'c', 'c','c'],

                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

data
#calculate means of each group

data.pivot_table(values='ounces',index='group',aggfunc=np.mean)
#calculate count by each group

data.pivot_table(values='ounces',index='group',aggfunc='count')