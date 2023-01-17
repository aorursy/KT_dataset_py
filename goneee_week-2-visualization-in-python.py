# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_path='../input/US GDP.csv'

us_gdp = pd.read_csv(data_path)

us_gdp
us_gdp.head(3) # show first 3 records
us_gdp.tail(3) # show last 3 records
us_gdp.iloc[32:34]
us_gdp.loc[33]
us_gdp.loc[32:34]
us_gdp.plot(kind='scatter',

           x='Year', y='US_GDP_BN',

           title='US GDP per year')
us_gdp.Year.min()

us_gdp['Year'].min()

#print("")


axes= us_gdp.plot(kind='line',x='Year', y='US_GDP_BN')

us_gdp.plot(kind='scatter',x='Year', y='US_GDP_BN',ax=axes,

           figsize=(12,5))

plt.title("From %d to %d" % (us_gdp['Year'].min(), 

                                us_gdp['Year'].max()))

plt.ylabel("GDP") # Changed ylabel from 'US_GDP_BN' to 'GDP'



plt.suptitle("      US GDP per year",size=15) # Change title
us_gdp.plot(kind='barh',x='Year')
us_gdp['GDP_Growth_PC'].plot(kind='bar')
us_gdp['GDP_Growth_PC'].value_counts()
gdp_growth_pc_binned = pd.cut(us_gdp['GDP_Growth_PC'],5)

gdp_growth_pc_binned
gdp_growth_pc_binned.value_counts()
# Bin our data set based on the values for year and assign

# labels

us_gdp_year_groups = pd.cut(us_gdp['Year'],3,

                            labels=['Early 1990s',

                                   'Late 1990s to Early 2000s',

                                   'After Early 2000s'])

print("="*32)

print(" Distribution per groups")

print("="*32)

# count the number of observations for each unique value

year_counts = us_gdp_year_groups.value_counts()

# rename series object for presentation in column

year_counts.name = 'Amount in each group'

pd.DataFrame(year_counts) # print data frame using jupyter
# Create a new data frame with the time period and and gdp

# by passing a python object to the pd.DataFrame method

us_gdp_with_year_group_data = pd.DataFrame({

    'Time_Period':us_gdp_year_groups,

    'US_GDP_BN':us_gdp['US_GDP_BN']

})

# print first 3 values

us_gdp_with_year_group_data.head(3)
# create a boxplot to show distrbiution of data

# ask boxplot to divide our data into groups

us_gdp_with_year_group_data.boxplot(by='Time_Period')

plt.title('US GDP distribution in each time period')

plt.suptitle('')

plt.annotate('This is the smallest',(0.8,7000))

print('')
us_gdp_with_year_group_data.groupby('Time_Period').describe()
# create a boxplot to show distrbiution of data

# ask boxplot to divide our data into groups

us_gdp_with_year_group_data.groupby('Time_Period').boxplot()

plt.title('US GDP distribution in each time period')

plt.suptitle('')

plt.annotate('This is the smallest',(0.8,7000))

print('')
us_gdp_with_year_group_data.groupby('Time_Period').describe()
us_gdp_with_year_group_data.describe()