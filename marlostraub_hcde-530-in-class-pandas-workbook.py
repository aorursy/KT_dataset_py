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
df=pd.read_csv('/kaggle/input/wdi-data/WDIData_smaller.csv')
df.head(1) #here we are just profiling the data to take a look at it; this means just looking at it to try and make sense of it.
#if we changed this to df.head(5), we would get the first 5 rows
#this is not like an index where we start with 0, we start with 1 here
df['Country Name'].head() #here we just look at a given column.
#we can do multiple columns by separating wtih a column, with all of those items in an additional set of brackets (we are requesting a python list)
df[['Country Name', 'Indicator Name']].head()
#we are slicing here; we are looking at a slace of data
df[['Country Name', 'Indicator Name']].head(1000)
#the output below that shows the top 5 and the bottom 5 is unique to kaggle. There is code you can add to scroll, but need to look that up. 
df[['Country Name', 'Indicator Name']].tail() #will show you the bottom rows

two_col_df=df[['Country Name', 'Country Code']].head()
two_col_df
two_col_df.to_csv('two_col_df.csv') #write this data to a csv named this, which we can see in our output in kaggle
df.iloc[0:5,0:5] #iloc is a method (integer locate) and says pass a list of parameters which is why we use brackets
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html

smaller=df.iloc[0:1000,:] #get all of the columns of the first 1k rows, the colon means give me everything smaller
smaller

smaller.tail(10)
smaller.to_csv('WDIDatasmaller.csv') #create a csv out of this data and name it this
df.iloc[:,-1:] #show me all the rows, in the last column; i could make this -2 or -3 to show the last two or three columns
#we want to look at everything for the USA, the equivalent of filtering in excel
#create a variable to hold our results 
usdf=df[df['Country Name']== 'United States']#wherever Country Name = United States
usdf.tail()#show the last 5 rows to see if this is working 
#because we have named this variable, we can now manipulate this as an object
usdf=usdf.drop('Unnamed: 62',axis=0) #we want to drop this unhelpful column. Specify the column name, and then axis=1 means we are telling it to drop the column