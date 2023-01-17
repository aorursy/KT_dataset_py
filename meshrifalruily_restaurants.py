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
# importing pandas library

import pandas as pd
#Importing Data (reading csv file)

df = pd.read_csv('../input/riyadhvenues/riyadh_venues.csv')
# Pandas Data Structures

df.head()
df.dtypes
x = df['name']

x
type(x)
x = df[['name']]

x
df.tail()
df.shape
df.size
df.info()
df.describe()
df.columns
# Using value_counts for knowing columns that contain strings / count of different values within a column

df['price'].value_counts()
#Selecting a column in a Data Frame (the resulting object will be  a Series)

df['name']
#Selecting a column in a Data Frame (the resulting object will be  a datafrme)

df[['name']]
#content of first 5 values of column 'name'

df[['name']][:5]
#Data Frames: Selecting rows (#Select rows by their position)

df[0:3]


df[['name', 'categories']].head()
#Data Frames: method loc

#Select rows by their labels:

df.loc[0:1]
#Data Frames: method loc

#Select rows by their labels:

df.loc[0:1, ['name', 'categories']]
# Data Frames: method iloc

# #Select rows by their poisitions:

df.iloc[0:2, 0:2]
df.head()
df.iloc[1:4, [0,1,5]]
df.loc[1:4, ['name','categories','users']]
df.head(6)
## iloc[row slicing, column slicing] (1st and 6th rows and 2nd and 4th columns)

df.iloc[[0,5], [0,4]]  
# First column

df.iloc[:, 0:2]
#First row of a data frame

df.iloc[0:1]
#First row of a data frame

df.iloc[0]


df['users'].max()
max(df['users'])
best_restaurant = df[df['users'] == df['users'].max()]
best_restaurant
#Be Careful 

df['rating'] == df['rating'].max()
# Best five restaurants in Riyadh

df.sort_values('rating', ascending = True).head(5)
#Data Frames: Sorting

df_sorted = df.sort_values(by= 'rating')

df_sorted
biggest_checkins = df[df['checkins']==67445]
biggest_checkins['name']
biggest_checkins['name'].iloc[0]


df[df['rating']> 8]
# Most common resturant 

df['categories'].value_counts()

# generate 'categories' DF

categories= pd.DataFrame(df.categories.value_counts().reset_index())

categories = ['categories', 'count']

brands
# Common Restaurant based on price 

df[df['price']== 'Cheap']['categories'].value_counts().head(10)
# Common Cheap Restaurant with hight rating

df[(df['rating']> 6) & (df['price'] == 'Cheap')].head()