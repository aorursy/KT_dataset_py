# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # matplotlib pyplot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dfo = pd.read_csv('../input/cleaned_generation.csv') # Import data file
dfo.head(4)
# Data Cleaning
dfo = dfo.loc[dfo['Region'].isin(['England', 'Northern Ireland','Scotland','Wales','Other Sites4'])]
dfo.replace('-', 0, inplace=True) # Clean up data
dfo.replace(',', '', regex=True) # Remove comma from numerical values
dfo.head(4)
# Here we'll assign the regions a numerical value
dfo = dfo.replace({'Region' : {
    'England' : 1,
    'Northern Ireland' : 2,
    'Scotland' : 3,
    'Wales' : 4,
    'Other Sites4' : 5
}})
df = dfo.apply(pd.to_numeric, errors="coerce") # Convert df object to integer
df.head(12)
# Analyse total output
total_output = df[['Year', 'Total']] # Separate total output from the rest of the data
total_output = total_output.groupby('Year').sum() # Group values by year and sum
# Plot results on a line graph
total_output.plot()
plt.ylabel('Energy Output (GWh)')
plt.legend(loc=1, prop={'size': 6})
plt.show()
df = df.drop(columns=['Unnamed: 0', 'Total']) # Remove Unnamed: 0, Total from the dataframe
df.fillna(0) # Replace any 'NaN' values
df.head(12)
# Firstly, we'll analyse the data by year
df_gy = df.drop(columns=['Region']) # Region isn't required for this analysis, so remove from df
group_year = df_gy.groupby('Year').sum() # Group by year and sum values
group_year.head(5)
# Plot results on a line graph
group_year.plot()
plt.ylabel('Energy Output (GWh)')
plt.legend(loc=1, prop={'size': 3})
plt.show()
# Next, we'll analyse the data by region
df_gr = df.drop(columns=['Year']) # Year not required for regional analysis, looking over whole period
group_region = df_gr.groupby([ 'Region']).sum() # Group by region and sum values
group_region.head(5)
# Plot bar chart of data