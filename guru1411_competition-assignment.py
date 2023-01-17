# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.columns
df.shape # Prints number of rows and columns in dataframe
df.head(10) # Prints first 10 rows of the DataFrame
df.tail(10) # Prints last 10 rows of the DataFrame
df.info() # Index, Datatype and Memory information
df.describe() # Summary statistics for numerical columns
df['first_active_month'].value_counts(dropna = False) # Views unique values and counts
df['feature_1'].value_counts(dropna = False) # Views unique values and counts
df['feature_2'].value_counts(dropna = False) # Views unique values and counts
df['feature_3'].value_counts(dropna = False) # Views unique values and counts
df['target'].mean() # Returns the mean of target column
df.corr() # Returns the correlation between columns in a DataFrame
df.count() # Returns the number of non-null values in each DataFrame column
df.max() # Returns the highest value in each column
df.min() # Returns the lowest value in each column
df.median() # Returns the median of each column
df.std() # Returns the standard deviation of each column
df['card_id'] # Returns column with label card_id as Series
df[['first_active_month', 'card_id']] # Returns Columns as a new DataFrame
df['card_id'].iloc[0] # Selection by position (selects first element)
df['first_active_month'].loc[0] # Selection by index (selects element at index 0)
df.iloc[0,:] # First row
df.iloc[0,0] # First element of first column
df.columns = ['First_Active_Month','Card_ID','Feature_A','Feature_B','Feature_C','Target'] # Renames columns
pd.isnull(df) # Checks for null Values, Returns Boolean Array
pd.notnull(df) # Opposite of s.isnull()
df.dropna() # Drops all rows that contain null values
df.dropna(axis=1) # Drops all columns that contain null values
df.dropna(axis=1,thresh=5) # Drops all rows have have less than 5 non null values
df.fillna(0) # Replaces all null values with 0
df['Target'] = df['Target'].fillna(df['Target'].mean())  # Replaces all null values with the mean (mean can be replaced with almost any function from the statistics section)
df[['Feature_A', 'Feature_B', 'Feature_C']].astype(float) # Converts the datatype of the series to float
tmp_df = df
tmp_df['Feature_A'].replace(1,'one') # Replaces all values equal to 1 with 'one'
tmp_df.rename(columns={'Target': 'Target_value'}) # Selective renaming
tmp_df.set_index('Card_ID') # Changes the index
df[df['Target'] > 0.5] # Rows where the target column is greater than 0.5
df[(df['Target'] > 0.5) & (df['Target'] < 0.7)] # Rows where 0.5 < target < 0.7
tmp_df.sort_values(['Target']) # Sorts values by target in ascending order
tmp_df.sort_values(['Target'],ascending=False) # Sorts values by target in descending order
tmp_df.sort_values(['First_Active_Month','Target'], ascending=[True,False]) # Sorts values by col1 in ascending order then col2 in descending order
tmp_df.groupby(tmp_df['First_Active_Month']).mean() # Returns a groupby object for values from one column
tmp_df
tmp_df.pivot_table(index='Card_ID', values= 'Target', aggfunc='mean') # Creates a pivot table that groups by col1 and calculates the mean of col2
tmp_df.groupby('First_Active_Month').agg(np.mean) # Finds the average across all columns for every unique column 1 group