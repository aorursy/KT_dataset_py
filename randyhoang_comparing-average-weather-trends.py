# Import libraries for data manipulation and data visulization
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame
%matplotlib inline
# Read the csv files and store the data into two separate dataframes
sanjose_df = pd.read_csv('../input/sanjose_data.csv', index_col=None)
global_df = pd.read_csv('../input/global_data.csv', index_col=None)
# Let's see how many rows and columns we are dealing with
sanjose_df.shape
# Let's see how many rows and columns we are dealing with
global_df.shape
# Grabbing more information about the two datasets
sanjose_df.info()
global_df.info()
# Taking a look at the first 5 rows of sanjose_df to see what year the data begins with
sanjose_df.head()
# Taking a look at the first 5 rows of sanjose_df to see what year the data ends with
sanjose_df.tail()
global_df.head()
global_df.tail()
# Make sure global_df is a dataframe
global_df = DataFrame(global_df)

# Eliminate all years that the San Jose data doesn't have
global_df = global_df[global_df.year >= 1849]
global_df = global_df[global_df.year <= 2013]

# Reset the index of the new dataframe and drop the potential new index row
global_df = global_df.reset_index(drop=True)

# Show the new DataFrame to make sure it works
global_df.head()
global_df.tail()
# Create a rolling mean and save that into a new variable called sanjose_decade and 
sanjose_new = sanjose_df['average_fahrenheit'].rolling(window=5).mean()
global_new = global_df['average_fahrenheit'].rolling(window=5).mean()

# Create a variable that holds both of the new rolling mean datasets 
result = pd.DataFrame({'San Jose': sanjose_new,
                       'Global': global_new})

# Plot the line graph
plt.plot(sanjose_df['year'],result)

plt.title('San Jose Average Temperature compared to Global Average Temperature')
plt.xlabel('Year')
plt.ylabel('Fahrenheit')
plt.legend(result)


