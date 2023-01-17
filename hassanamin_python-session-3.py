# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



plt.style.use('ggplot')
# Importing Essential Packages

import numpy as np

import pylab as pl

# Printing Numpy Version

print("Numpy Version : ",np.__version__)
# Simple Line Plot

year = [1950, 1970, 1990, 2010]

pop = [2.519, 3.692, 5.263, 6.972]

plt.plot(year, pop)

#Show

plt.show()
# Simple Line Plot with Axis Labels and Title

import matplotlib.pyplot as plt

year = [1950, 1970, 1990, 2010]

pop = [2.519, 3.692, 5.263, 6.972]

plt.plot(year, pop)

#Adding Axis Labels

plt.xlabel('Year')

plt.ylabel('Population')

plt.title('World Population Projections')

#Show

plt.show()
# Adding Ticks to above Graph

# Simple Line Plot with Axis Labels and Title

import matplotlib.pyplot as plt

year = [1950, 1970, 1990, 2010]

pop = [2.519, 3.692, 5.263, 6.972]

plt.plot(year, pop)

#Adding Axis Labels

plt.xlabel('Year')

plt.ylabel('Population')

plt.title('World Population Projections')

# plt.yticks([0,2,4,6,8,10])

#Show

plt.show()
import matplotlib.pyplot as plt

year = [1950, 1970, 1990, 2010]

pop = [2.519, 3.692, 5.263, 6.972]

plt.scatter (year, pop)

plt.show()

import matplotlib.pyplot as plt

values = [0,0.6,1.4,1.6,2.2,2.5,2.6,3.2,3.5,3.9,4.2,6]

plt.hist(values, bins = 3)

plt.show()

d = {'user':'bozo', 'pswd':1234}

print(d['user'])

# Removing one element

del(d['user'])

print(d)

# Removing all elements

d.clear()

print(d)

#adding a new element

d['id'] = 45

print(d)

d = {'user':'bozo', 'p':1234, 'i':34}

# List of keys.

print('List of keys \n ',d.keys()) 

# List of values.

print('List of values \n ',d.values()) 

# List of item tuples.

print('List of item tuples \n ',d.items()) 

import pandas as pd

import numpy as np

# Create a dataframe

raw_data = {'first_name': ['Jason', 'Molly', np.nan, np.nan, np.nan], 

        'nationality': ['USA', 'USA', 'France', 'UK', 'UK'], 

        'age': [42, 52, 36, 24, 70]}

df = pd.DataFrame(raw_data, columns = ['first_name', 'nationality', 'age'])

print(df)
# Pre-defined lists

names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']

dr =  [True, False, False, False, True, True, True]

cpc = [809, 731, 588, 18, 200, 70, 45]



# Import pandas as pd

import pandas as pd



# Create dictionary my_dict with three key:value pairs: my_dict

my_dict={'country':names,'drives_right':dr,'cars_per_cap':cpc}



# Build a DataFrame cars from my_dict: cars

cars= pd.DataFrame(my_dict)



# Print cars

print(cars)

import pandas as pd

# Build cars DataFrame

names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']

dr =  [True, False, False, False, True, True, True]

cpc = [809, 731, 588, 18, 200, 70, 45]

dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }

cars = pd.DataFrame(dict)

print(cars)

# Definition of row_labels

row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars

cars.index=row_labels

# Print cars again

print(cars)

# Import pandas as pd

import pandas as pd



# Fix import by including index_col

cars = pd.read_csv('../input/basics/cars.csv',index_col=0)



# Print out cars

print(cars)

import pandas as pd

brics = pd.read_csv('../input/basics/brics.csv', index_col = 0)

print(brics)

print('Country Column Type as Series : ',type(brics['country']))

print('Country Column Type as Dataframe : ',type(brics[['country']]))

print('Country, Capital as Dataframe :\n ',brics[['country','capital']])

print('Row Access :\n ',brics[1:4])
# Row Access using loc

print('Row Access using loc\n')

print(brics.loc[["RU", "IN", "CH"]])

print('\n Column access \n') 

print(brics.loc[:, ["country", "capital"]])

print('\n Row & Column access \n') 

print(brics.loc[["RU", "IN", "CH"], ["country", "capital"]])
print('Row access : \n') 

print(brics.iloc[[1]])

print('\n Multiple Rows \n',brics.iloc[[1,2,3]])

print('\n Row & Column access : \n')

print(brics.iloc[[1,2,3], [0, 1]])
print("Retrieving all the rows and some columns")

brics.iloc[:,[1,2]]
import pandas as pd

import numpy as np



dates = pd.date_range('1/1/2000', periods=8)

print(dates)

df = pd.DataFrame(np.random.randn(8, 4),index=dates, columns=['A', 'B', 'C', 'D'])

print(df)
s = df['A']

s[dates[5]]
df[['B', 'A']] = df[['A', 'B']]
a = pd.Index(['c', 'b', 'a'])

b = pd.Index(['c', 'e', 'd'])

print(a|b)

print(a&b)

print(a.difference(b))
s.iat[5]
df.at[dates[5],'A']
import pandas as pd



l_1d = [0, 1, 2]



s = pd.Series(l_1d)

print(s)



s = pd.Series(l_1d, index=['row1', 'row2', 'row3'])

print(s)
l_2d = [[0, 1, 2], [3, 4, 5]]



df = pd.DataFrame(l_2d)

print(df)



df = pd.DataFrame(l_2d,

                  index=['row1', 'row2'],

                  columns=['col1', 'col2', 'col3'])

print(df)
import pandas as pd

iris_data = pd.read_csv("../input/iris-dataset/iris.data.csv")
#import the pandas library and aliasing as pd

import pandas as pd

s = pd.Series()

print(s)
#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

data = np.array(['a','b','c','d'])

s = pd.Series(data)

print(s)
#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

data = np.array(['a','b','c','d'])

s = pd.Series(data,index=[100,101,102,103])

print(s)
#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

data = {'a' : 0., 'b' : 1., 'c' : 2.}

s = pd.Series(data)

print(s)

#import the pandas library and aliasing as pd

import pandas as pd

import numpy as np

s = pd.Series(5, index=[0, 1, 2, 3])

print(s)
import pandas as pd

s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])



#retrieve the first element

print(s[0])
import pandas as pd

s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])



#retrieve the first three element

print(s[:3])
import pandas as pd

s = pd.Series([1,2,3,4,5],index = ['a','b','c','d','e'])



#retrieve the last three element

print(s[-3:])
# importing pandas module  

import pandas as pd  

  

# importing regex module 

import re 

    

# making data frame  

data = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/nba.csv")  

    

# removing null values to avoid errors  

data.dropna(inplace = True)  

  

# storing dtype before operation 

dtype_before = type(data["Salary"]) 

  

# converting to list 

salary_list = data["Salary"].tolist() 

  

# storing dtype after operation 

dtype_after = type(salary_list) 

  

# printing dtype 

print("Data type before converting = {}\nData type after converting = {}".format(dtype_before, dtype_after)) 

  

# displaying list 

salary_list 
import re



txt = "The rain in Spain"

x = re.search("^The.*Spain$", txt)

print(x)
import re



txt = "The rain in Spain"

x = re.findall("ai", txt)

print(x)

import re



txt = "The rain in Spain"



#Check if the string starts with "The":



x = re.findall("\AThe", txt)



print(x)



if (x):

  print("Yes, there is a match!")

else:

  print("No match")

import re



txt = "The rain in Spain"



#Check if the string contains any digits (numbers from 0-9):



x = re.findall("\d", txt)



print(x)



if (x):

  print("Yes, there is at least one match!")

else:

  print("No match")
