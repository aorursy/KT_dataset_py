#importing the necessary libs.

import pandas as pd

import numpy as np

#Step 2. Import the dataset from this address.

#Assign it to a variable called users and use the 'user_id' as index



users = pd.read_table('../input/occupation.txt', sep='|', index_col='user_id')
#see the first 5 entries.

users.head(5)
#see the last five entries

users.tail(5)
#What is the number of observations in the dataset?

users.shape[0]
#What is the number of columns in the dataset?

users.shape[1]
#Print the name of all the columns.

users.columns
#How is the dataset indexed?

users.index
#What is the data type of each column?

users.dtypes
#Print only the occupation column

users.occupation

users['occupation'].head(1)
#How many different occupations there are in this dataset?

users.occupation.nunique()

#Alternatively

users.occupation.value_counts().count()
# What is the most frequent occupation?

users.occupation.value_counts().head(3)
#Summarize the DataFrame.

users.describe() #retuns only the numric cols
#Summarize all the columns

users.describe(include = 'all') 
# Summarize only the occupation column

users.occupation.describe()
#What is the mean age of users?

round(users.age.mean())
#What is the age with least occurrence?

users.age.value_counts().tail()