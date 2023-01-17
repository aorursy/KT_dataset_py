import pandas as pd
import numpy as np

ds = pd.read_csv('../input/ds1.csv')
ds.head(5)
ds = pd.read_csv('../input/ds1.csv')
ds.columns
ds.describe()
ds

# Or you can use the '[]' notation to display all columns. Uncomment the below line to see results.
# ds[:]
# Any of the following scripts will show all data, since we are selecting all rows and all columns

ds.iloc[:,:]

ds.loc[:,:]
# Using head() function
ds.head(10)

# Using the [] notation - [] notation is used to either select a range of rows (as in the following script) or to select a specific column
# ds[:10] 
ds[0:10]

# Using iloc - Select rows from 0 to 10
# ds.iloc[:10]
ds.iloc[0:10]
ds.sample(10)
# Retrieving a single column

# Returns a Series for first 10 E Mail addresses
ds['E Mail'][:10]

# Returns a DataFrame for first 10 E Mail addresses
ds[['E Mail']][:10]

# Using iloc 
# Note the method used to retrieve the columns position of 'E Mail'.
# Since the second parameter will only accept positions of the columns, the method was used.
# Returns a DataFrame
ds.iloc[:10,[ds.columns.get_loc('E Mail')]]

# Using loc
# The below method will only work if the index has been set to the positional values of the rows.
# In case it is something else, uncomment the following line to reset the index to positional values.

# ds.reset_index(drop=True, inplace=True)
ds.loc[:10,['E Mail']]
# Retrieving multiple columns

# Using iloc - Select the 6th and 5th columns which are 'E Mail' and 'Gender' respectively
ds.iloc[:10,[6,5]]

# Using loc - Select the columns 'E Mail' and 'Gender' by mentioning the label names explicitly
ds.loc[:10,['E Mail', 'Gender']]

# Conversely, we can extract the all the rows by pulling out the required columns from the DataFrame and then select the first 10 rows.
ds[['E Mail', 'Gender']][:10]
insert_val = pd.DataFrame([['633433', 'Dr.', 'Spider', 'H', 'Man', 'M', 'spider.man@gmail.com', 'Richard Parker', 'Mary Parker', 'Mary', 500000]], columns=ds.columns)
ds = ds.append(insert_val, ignore_index=True)

# The parameter ignore_index=True ensures that the indexing is done automatically. If you have an indexing already defined, make it False.
insert_vals = pd.DataFrame([['633433', 'Dr.', 'Spider', 'H', 'Man', 'M', 'spider.man@gmail.com', 'Richard Parker', 'Mary Parker', 'Mary', 500000], [633434, 'Dr.', 'Iron', 'H', 'Man', 'M', 'iron.man@gmail.com', 'Howard Stark', 'Maria Stark', 'Maria', 7500000]], columns=ds.columns)
ds = ds.append(insert_vals, ignore_index=True)
ds.drop([0,1], axis=0, inplace=True)

# axis=0 is for rows, axis=1 is for columns,
# inplace=True to make the change permanent
ds.drop(ds[ds['Middle Initial']=='A'].index, axis=0)
ds.loc[ds['Gender']=='M',['Gender']]='Male'
ds.loc[ds['Gender']=='Male',['Gender', 'Name Prefix']]='M','Doctor'
ds.count()
ds['Salary'].astype(int).sum()
ds['Salary'].astype(int) + 25
average = ds['Salary'].astype(int).mean()
average
ds['Gender'].unique()
ds['Gender'].value_counts()
ds.sort_values('Middle Initial', ascending=True)