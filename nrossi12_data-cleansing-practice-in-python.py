# Importing libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Import data and convert time columns from string to timestamp

df = pd.read_csv('../input/startup-investments-crunchbase/investments_VC.csv', encoding = "ISO-8859-1", parse_dates=['founded_month', 'founded_quarter', 'founded_year'])
# Strip empty spacing from column headers for easier column calling

df.rename(columns=lambda x: x.strip(), inplace=True)
df.head()
df.tail()
df.isnull().sum()
# Shape of data before removing NaN rows

df.shape
# Remove any row with NaN as ALL column values

# (Calling the data base affected by this command by a new name just in case there is an error)

data = df.dropna(how='all')
# Shape of data after removing NaN rows

data.dropna(how='all').shape
54294 - 49438
data = data.dropna(how='any')
data.shape
pd.set_option('display.max_columns', 40)

data.head()
# Some columns are strings, some numeric and some time valued. Timestamp was already done on importing so now to make

# sure that other columns are correct data types. Numeric columns will be converted into float data types.

data.dtypes
# Remove all commas in funding_total_usd column

data['funding_total_usd']= data['funding_total_usd'].str.replace(',', '')

data['funding_total_usd']= data['funding_total_usd'].str.replace('-', '0')
data.head()
# convert whole column funding_total_usd to float data type

data['funding_total_usd'] = data['funding_total_usd'].astype(float)
data.head()
data[['state_code', 'region', 'city']]= data[['state_code', 'region', 'city']].astype(str)
state_count = data['state_code'].value_counts()
print(f"Minimum = {np.min(state_count)}")

print(f"Maximum = {np.max(state_count)}")

print(f"Median = {np.median(state_count)}")

print(f"Mean = {round(np.mean(state_count), 2)}")

print(f"Standard deviation = {round(np.std(state_count), 2)}")

print(f"Variance = {round(np.var(state_count), 2)}")
state_count1 = state_count[:29,]

state_count2 = state_count[30:,]



plt.figure(figsize=(20, 10))

plt.subplot(211)

plt.plot(state_count1, color='tab:blue', marker='o')

plt.plot(state_count1, color='black')



plt.subplot(212)

plt.plot(state_count2, color='tab:blue', marker='o')

plt.plot(state_count2, color='black')



plt.show()