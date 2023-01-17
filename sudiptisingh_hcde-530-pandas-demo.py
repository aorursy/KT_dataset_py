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
#df is variable for dataframe

#pd is the function module

df = pd.read_csv('../input/wdi-data/WDIData_smaller.csv')
df.head(1)
#to find content of only this column

df['Country Name'].head()
#to find content of two columns

#this is called slicing data

#add a number like head(100) to see the first 100 rows

df[['Country Name', 'Country Code']].head()
#if you wanted to only look at the back-end of this data, you could use tail

df[['Country Name', 'Country Code']].tail()
#create a new variable to plug this in elsewhere

two_col_df = df[['Country Name', 'Country Code']].tail()

#run that command

two_col_df
#convert to csv file

two_col_df.to_csv('two_col_df.csv')
#to only see a range of [rows, columns]

df.iloc[0:5,0:5]
#if you want to see the first 1000 rows, but all the columns, just use a colon

smaller = df.iloc[0:1000,:]

smaller
#if you wanted to see the last ten

smaller.tail(10)
smaller.to_csv('WDIDatasmaller.csv')
#i want all the rows, and everything in the last column

df.iloc[:,-1:]
#find only where country name is United States

usdf = df[df['Country Name'] == 'United States']

#this variable lets you play around with data related to usdf

usdf.tail()
#to remove a row use drop

#to do a column you must specify axis as 1 since the vertical element is 1

#you can only run this once, and then it may show an error?

usdf = usdf.drop('Unnamed: 62', axis=1)
#run it to observe the drop

usdf.head()
#convert this data to json file, refer to this on the righthand side under kaggle/working

usdf.to_json('filtered_WDIData.json')