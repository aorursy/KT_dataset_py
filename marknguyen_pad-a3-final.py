import pandas as pd

import numpy as np
# Import walmart data

walmart_df = pd.read_csv("../input/walmart.csv")

print(walmart_df.columns)
# What does the DataFrame look like? Print out the first 5 and last 5 rows of data

print(walmart_df.head(5))

print(walmart_df.tail(5))
# What do the dates look like? What is the date range?  

## Hint: Use unique: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html

dates = walmart_df.Date.unique()



print(dates)

print(dates[:5])

print(dates[-5:])
# What are the data types for each column of data?

print(walmart_df.dtypes)
# Specifically what kind of objects does Date and Weekly_Sales represent?

print(type(walmart_df.Date.iat[0]))

print(type(walmart_df.Weekly_Sales.iat[0]))

print(type(walmart_df.IsHoliday.iat[0]))
# How many records are there for each department?

# Hint: Use the function value_counts: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html

walmart_df.Dept.value_counts()
# For each column of data, how many missing values are there?

walmart_df.isna().sum()
# Filter out rows of that data that have missing values in either Date or Weekly_Sales columns

clean_df = walmart_df.dropna()
# Confirm that all missing values are dropped

clean_df.isna().sum()