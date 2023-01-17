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
# Pandas is used for data manipulation and analysis

# First of all we need a dataframe to work on it and we need a dictionary to create a dataframe.

dict1 = {"Name":["James","Michael","Morgan","Barrack","Felipe","Pablo","Maria","Leonardo"],

       "Surname":["Milner","Jordan","Freeman","Obama","Luis","Escobar","Sharapova","Di Caprio"],

       "Age":[36,43,60,52,30,70,35,47],

       "Salar y":[3000,5000,4000,7500,2000,5300,4600,6000]}

df = pd.DataFrame(dict1)

print(df)

print(df.columns)



# IMPORTANT NOTE: We wrote "Salary" as "Salar y" to show some script differences. Please, just ignore it for now.
# Data overview

print(df.head()) # First 5 rows of data

print("-"*40)

print(df.tail()) # Last 5 rows of data

print(df.tail(2)) # Last 2 rows of data
# Information section about data

print(df.info()) # information about dataframe

print(df.describe()) # Statistical information about integer columns

print(df.columns.dtype) # Datatypes of columns
# Basic filtration section

print(df.Age)        # 1st way to print Age column

print(df["Age"])     # 2nd way to print Age column

print(df.iloc[:,2])  # 3rd way to print Age column

print(df.columns[1])

print("-"*50)

print(df.iloc[:,1:3])

print(df.iloc[:,1:])

print(df.loc[2:4,["Name","Salar y"]])

print(df.iloc[:,::-1]) # Reversed columns

print(df.iloc[::-1,:]) # Reversed rows

print(df.iloc[:,-1])   # Last integer column
# Detailed filtration section

filter1 = df.Age<45

filter2 = df["Salar y"]<4100

filtereddata = df[filter1&filter2]    # Combined filters

print(filtereddata)

# Let's check our filters' types

print(type(filter1))

print(type(filtereddata))
# Min, max, mean, count etc.

print(df.Age.mean())

print(df["Salar y"].mean())

print(df.Age.min())

print(df["Salar y"].count()) # Number of row
# Adding new columns

df1=df

df1["new_column"] = [11,12,13,14,15,16,17,18] # Add a new column

df1["new_column2"] = [i*2 for i in df1.Age] # First way to add new column which is connected to another column

def f(x):

    return x*2

df1["Apply_column"]=df.Age.apply(f) # Second way to add new column which is connected to another column

df1["filtered_column"] = ["Low" if i<4100 else "High" for i in df1["Salar y"]]

print(df1)

print("-"*60,"1")

print(df1.loc[:,["Age","new_column2"]])

print(df1.loc[:,["Salar y","filtered_column"]])
# Reorganize the columns' script

df1.columns = [i.lower() for i in df1.columns] # lower case

df1.columns = [i.split()[0]+"_"+i.split()[1] if len(i.split())>1 else i for i in df1.columns] # "salar y" is 2 words. We made it 1 with "underline"

print(df1.columns)
# Drop

df2=df

print(df2.drop(["surname"],axis=1)) # We dropped surname column TEMPORARILY

print("-"*60,"1")

print(df2) # As you see, surname column is still there

df2.drop(["surname"],axis=1,inplace=True)

print("-"*60,"2")

print(df2)
# Concatenating 2 data sets

df3 = df

concat1 = df3.head(2)

concat2 = df3.tail(2)

concatenatedvertical = pd.concat([concat1,concat2],axis=0)

concatenatedhorizontal = pd.concat([concat1,concat2],axis=1)

print(concatenatedvertical)

print("-"*60,"1")

print(concatenatedhorizontal)
# As a datascience candidate, I applied what I've learnt about pandas and data science in this Kernel.

# Waiting for your supports and feedbacks.



# Thank you for your time.