# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Famous manned submersible ALVIN's dive log
data = pd.read_csv('../input/alvin_raw.csv')
# We can see the last dive is on the top of the dive log.
data.head()
# And the first dive is on the bottom of the dive log.
data.tail()
# to see number of rows and columns
data.shape
# To index the first dive (gives us the index number which is 4931)
data['Dive'][1]
# The same thing as above
data.Dive[1]
# To use loc accessor (to see last dive's depth)
data.loc[0,'Depth']
# Selecting only columns names; dive, depth and date.
data[['Dive','Depth','Date']].head(10)
# Visualization of depths from first to last dive
data1 = data[['Dive','Depth','Date']]
data1.plot(kind='scatter', x='Dive', y='Depth', alpha=0.2, figsize=(10,6))
plt.xlabel('Number of the Dive')
plt.ylabel('Depth of the Dive (meters)')
plt.title('ALVIN\'s Dive Depths')
plt.show()
# Slicing Data Frames from 1st to 15th indexes
data.loc[1:15, 'Date':'Depth']
# Reverse slicing from the last index to 10th from the bottom 
data.loc[:4921:-1, 'Date':'Depth']
# Filtering Data Frames
# Creating Boolean series
boolean = data1.Depth > 4400
print(data1[boolean])
print(data1[boolean].shape)
# Combining Filters
# Filtering dataframe to see Alvin's dives deeper than 1000 m at her first 100 dives
filter1 = data1.Depth > 1000
filter2 = data1.Dive <= 100
print(data1[filter1 & filter2])
# Filtering column based 
# Filter the dives deeper than 4400 meters and show us the number of those dives
data1.Dive[data1.Depth>4400]
# Transforming Data
# Transforming the most recent dives number as 1st and the last one as 4932
def reverse_dive(n):
    return 4932-n
data.Dive = data.Dive.apply(reverse_dive)
data.head()
# Or we can use lambda function
# Turning dive numbers back to beginning
data.Dive = data.Dive.apply(lambda n : 4932-n)
data.head()
# Defining a new column by using other columns
# We can combine observer 1 and 2 together in a new column
data['Observers'] = data['Obs 1'] + ', ' +data['Obs 2']
data.tail()
# To see the index name
print(data.index.name)
# Now we can change it to Index_Number
data.index.name = 'Index_Nu'
data.head()
# To set the column name '0' as the index 
data.index = data['0']
data.head()
# Hierarchical indexing
# Setting index Cruise and Ops Area
data2 = data.set_index(['Cruise', 'Ops Area'])
data2.head(100)
# Pivoting Data Frames
# lets create a new dataframe
data3 = data1.head()
data3
# pivoting the dataframe
# New data frame has Dive numbers as indexes, dates as columns and depths as values
data3.pivot(index='Dive', columns = 'Date', values = 'Depth')
data3
# Stacking Data Frames
data4 = data3.set_index(['Dive', 'Date'])
data4
# Unstacking data frames
data4.unstack(level=0)
# Unstacking
data4.unstack(level=1)
data4
# Changing inner and outer indexes
data5 = data4.swaplevel(0,1)
data5
data3
# Melting Data Frames
pd.melt(data3, id_vars='Date', value_vars=['Dive', 'Depth'])
# Categorical and Groupbye
# Lets use our first data frame and group it with operation area and show us the max depths
data.groupby('Ops Area').Depth.max()
# Now group the same data frame with operations area and show us the mean of the depths
data.groupby('Ops Area').Depth.mean()
# Group the same data frame with operations area and show us min depths and dive number
data.groupby('Ops Area')[['Dive', 'Depth']].min()
