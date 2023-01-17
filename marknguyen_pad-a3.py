import pandas as pd

import numpy as np
# Import walmart data

walmart_df = pd.read_csv("../input/walmart.csv")

print(walmart_df.columns)
# What does the DataFrame look like? Print out the first 5 and last 5 rows of data

# What do the dates look like? What is the date range?  

## Hint: Use unique: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.unique.html

# What are the data types for each column of data?

# Specifically what kind of objects does Date and Weekly_Sales represent?

# How many records are there for each department?

# Hint: Use the function value_counts: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.value_counts.html

# For each column of data, how many missing values are there?

# Filter out rows of that data that have missing values in either Date or Weekly_Sales columns

# Confirm that all missing values are dropped
