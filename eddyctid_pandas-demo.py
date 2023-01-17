import pandas as pd
# Read Data from CSV file to DataFrame
df = pd.read_csv('../input/death.csv',index_col='Year')
# Selection using pandas
df

# view first 10 rows
df.head(10)
# Select 1999th Row
df.iloc[1999]
# Select all rows with index == 1999
df.loc[1999]
# Select multiple rows by Location
df.iloc[[1,2,3]]
# Select all rows with Index 1999 & 2011
df.loc[[1999,2011]]
# Slicing using pandas
# Slice from 0th to 11th Row
df[:11]
df[3:11]
#  Indexing in pandas
df.Deaths.head()
# df.Cause Name.head() won' work as there is Space between words
# To solve this kind of Problem we have to rename Columns
df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)
df
df
# Now we have to remove '-' symbol from 'age-adjusted_death_rate'
df.columns = [col.replace('-', '_').lower() for col in df.columns]
print(df.columns)
df
# Indexing: Columns & Rows
df.loc[2011][['state','cause_name','deaths']].head()
# Boolean Indexing
(df.state == 'Alabama').head()
# Selective Search
df[df.state == 'Alabama'].deaths
# Multiple Argument
df[(df.state == 'Alabama') & (df.cause_name == 'Cancer') & (df.deaths > 10000)].head(10)
# Using lambda expression
# Find state with 2 worded names
df[df['state'].str.split().apply(lambda x: len(x) == 2)].loc[2011].state.head(20)
# .isin
df[df.state.isin(['Alabama', 'New York', 'California'])].loc[2013].head(20)

