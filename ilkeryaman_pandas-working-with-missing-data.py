import numpy as np
import pandas as pd
arr = np.array([[10, 20, np.nan], [5, np.nan, np.nan], [21, np.nan, 10]]) # np.nan means not a number value

arr
df = pd.DataFrame(arr, index=["Index1", "Index2", "Index3"], columns=["Column1", "Column2", "Column3"])

df
"""
dropna drops columns/rows that contains missing values.

Since it is dropping according to axis and default value = 0 (x), it will drop all columns
"""
df.dropna() 
df.dropna(axis=1) # Since axis value = 1 (y), only Column2 and Column3 are dropped. 
df.dropna(thresh=2) # If there are at least 2 not a number value, do not remove this index
df.fillna(value=1) # Fill missing values with 1
df.sum() # Get sum, group by column.
df.sum().sum() # Get sum of all dataframe
df.fillna(value=df.sum()) # Fill missing values with sum of related column.
df.fillna(value=df.sum().sum()) # Fill missing values with sum of all frame.
df.size # size of dataframe
df.isnull() # Show as True/False
df.isnull().sum() # Column-based count of missing values
df.isnull().sum().sum() # Count of missing values at dataframe
def calculate_mean_of_dataframe(df):
    total_sum = df.sum().sum()
    total_num = df.size - df.isnull().sum().sum()
    return total_sum / total_num
calculate_mean_of_dataframe(df)
df.fillna(value=calculate_mean_of_dataframe(df)) # Fill missing values with mean of dataframe