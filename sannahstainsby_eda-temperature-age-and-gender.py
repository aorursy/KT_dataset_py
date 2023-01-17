# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import ast, json

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv')
# Take closer look at data set 

data.describe()
# Examine unique values in the data set.

data.apply('nunique')
data.columns
data = data.drop((['Publication (Citation)', 'Data contributor', 'Ta_h (F)', 'Air temperature (F)','Ta_m (F)', 'Ta_l (F)', 
                  'Operative temperature (F)', 'Radiant temperature (F)', 'Globe temperature (F)', 'Tg_h (F)', 'Tg_m (F)',
                  'Tg_l (F)', 'Air velocity (fpm)', 'Velocity_h (fpm)', 'Velocity_m (fpm)', 'Velocity_l (fpm)',
                   'Outdoor monthly air temperature (F)', ]), axis=1)
# Duplicates

print(data.shape)

duplicate_rows_data = data[data.duplicated()]

print(duplicate_rows_data.shape)
# Remove duplicated data 

data = data.drop_duplicates(keep='first')

print(data.shape)
# Check what type of data we have in our data set.

data.dtypes
# Transform datapoints / values that are registered as numbers (float64) despite being categorical (e.g.y|n door, window etc.)

data = data.copy() 

data["Heater"]= data["Heater"].astype("object")
data["Door"]= data["Door"].astype("object")
data["Window"]= data["Window"].astype("object")
data["Fan"]= data["Fan"].astype("object")
data["Blind (curtain)"]= data["Blind (curtain)"].astype("object")
data["Humidity sensation"]= data["Humidity sensation"].astype("object")
data["Air movement preference"]= data["Air movement preference"].astype("object")

# May need to transform more, need list of data, e.g. idk meaning of PPD.

# Check data points to check if transformation was successful. 

data.dtypes


# Look at relationship between gender and said categorical values through data visualization 

sns.regplot(x="Sex", y="tip", data=tips, color=".3");
# Check for missing values (data quality) to drop more columns that have missing values and reduce noise.  

print(data.isnull().sum())
# drop empty columns 

data = data.drop((['Koppen climate classification', 'Climate', 'Country', 'Database']), axis=1)
# get overview of data - easy to see where major and minor gaps are in your data

import missingno as msno

msno.matrix(data.select_dtypes(include='number'));

# results: Data is not of the best quality and there are significant gaps in many of the columns 
# Examine relationship between the different varibales and gender 

data.corr()['Sex'].sort_values(ascending=False).head(10)
# check out 'heater' (has the most outliers) using the describe() function
data["Heater"].describe()
data["Age"].describe()

# The average age of participants in the data set is 32 years, oldest is 99 years, and youngest is 6 years. 
# The majority of participants (75%) were 43 years and older.
data["Thermal sensation"].describe()

# Variance between sensations were 0.15 degrees celcius on average, 
data["Year"].describe()

# Most of the buildings are built within the last 10 years (75% after 2011) 
# with the 'youngest' building built in 2016 and the 'oldest' in 1979.
# In other words, all buildings are fairly new. 