# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the csv and describe the information about columns

global_temp_data = pd.read_csv('/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv')
df = pd.DataFrame(global_temp_data)

print ("Dataset : Information about columns: \n")
df.info()
#Access the first index.

df.iloc[0]
# Accessing Temparetures for Pune City.

df.loc[df.City == 'Pune']
# Finding the place with maximum recorded temperature.

df.loc[df['AverageTemperature'].idxmax()]
# Indexing the values as Country and City

df = df.set_index(['Country','City'])
df.head()
# Displaying Temperatures for the location Pune,India.
df.loc['India','Pune']['AverageTemperature']
# Reset index for futher operations.

df = df.reset_index()
#Country wise temperature.

df[['Country','AverageTemperatureUncertainty']].head(20)
# Reset index for futher operations.

df = df.reset_index()
# Show places where temperature is 2.

country_set = df['AverageTemperature'] == 2
df[['Country','City','AverageTemperature']][country_set]
# Sorting the data according the temperature.

df.sort_values('AverageTemperature', ascending = False)
#Sorting by date.

df.sort_values('dt', ascending = True)
# Reset index for futher operations.

df = df.reset_index()
# Describe the column attribues.

df.columns
# Attribute of Index.

df.index
# Datatype of all column values.

df.dtypes
# Count temperature records.

df.AverageTemperature.value_counts()  
# Unique values of recorded temprature.

df.AverageTemperature.nunique()
# Unique attributes of column City.

print('Unique Cities in the dataset')
df['City'].unique()
#Format to display temperature.

df['AverageTemperature']
#Format to display Country.

df['Country']
# Setting dt column to the date datatype
df.astype({'dt':'datetime64'}).head()
# Round the decimal places.

df.round(decimals=1).head(20)
# Indetify the missing values.

df.isnull()
# Replace Nan or null values with string to state temperature was not recorded.

df["AverageTemperature"].fillna("Not Recorded", inplace = True) 
df["AverageTemperatureUncertainty"].fillna("Not Recorded", inplace = True)
df.head(20)