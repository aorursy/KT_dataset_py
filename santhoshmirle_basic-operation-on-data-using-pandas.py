

import numpy as np 

import pandas as pd

import os



# Any results you write to the current directory are saved as output.

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



# Display Data input_data read from the 'train.csv' file

train_data


# Get the Share (Rows and Columns) on the table

train_data.shape
# Diaply First 5 Rows of data including Columns

train_data.head()
# Diaply First 10 Rows of Data Including Columns

train_data.head(10)
# Display Last 5 Rows of Data including Headers

train_data.tail()
# Disply Last 10 Rows of Data Including Columns

train_data.tail(10)
# Get a concise summary of the dataframe

train_data.info() 
# Describe is used to view some basic statistical details like percentile, mean, std etc. 

# of a data frame or a series of numeric values.

train_data.describe()
# Finds is there any null items on the Dataframe

train_data.isnull().any()
# Get All the Columns on the DataFrame

train_data.columns[:]
# Find the Null values on the DataFrame Columns wise

train_data.isnull().any(axis=0)
# Print out Name column as pandas series

train_data['Name'].head()
# Print out 'Name' and 'Age' column as pandas series

train_data[['Name','Age']].head()
# Display the row item on the data frame where Name is 'Heikkinen, Miss. Laina'

train_data[train_data['Name'] == 'Heikkinen, Miss. Laina']
# Diaply the item by Sorting based on Alphabests on Column 'Name'

train_data.sort_values('Name').head()
# Diaply the item by Sorting based on Alphabests on Column 'Age'

train_data.sort_values('Age').head()