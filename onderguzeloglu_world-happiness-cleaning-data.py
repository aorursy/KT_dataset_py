# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/world-happiness/2019.csv')
# head shows top 5 rows
data.head()
# tail shows last 5 rows
data.tail()
# columns gives names of features
data.columns
# shape gives number of rows and columns in a tuble
data.shape
# Info gives data type like dataFrame, number of sample or row, number of feature or column, feature types and memory usage
data.info()
#lets look frequency of world-happiness
print(data['Country or region'].value_counts(dropna = False)) # if there are nan values that also be counted
data.describe() # ignore null entries
# For example: compare  Freedom to make life choices  of world happiness that are   Social support or not
# black line at top is max
# blue line at top is 75%
# green line is median (50%)
# Blue line at bottom is 25%
# black line at bottom is min
data.boxplot(column = 'Generosity', by = 'Freedom to make life choices')
#Firstly create new data from world hapiness dta to explain melt nore easily.
data_new = data.head() # I only take 5 rows into new data
data_new
#lets melt
#id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame = data_new, id_vars = 'Country or region', value_vars=['Score','Social support'])
melted
# Index is name
# I want to make that columns are variable
# Finaly values in columns are value
melted.pivot(index = 'Country or region', columns = 'variable', values = 'value')
#Firstly lets create 2 data frame
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index = True) # axis = 0 : adds dataframes in row
conc_data_row
data.dtypes
# Lets convert object(str) to categorical 
# and int to float
data['Country or region'] = data['Country or region'].astype('category')
data['Generosity'] = data['Generosity'].astype('int')
data.dtypes
#Lets look at does world hapiness data have nan value
data.info()
#Lets check Generosity
data['Generosity'].value_counts(dropna = False)
# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Generosity"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?
#Lets check with assert statement
#Assert statement:
assert 1==1 # return nothing because it is true
#In order to run all code, we need to make this line comment
#assert 1==2 # return error because it is false
assert  data['Generosity'].notnull().all() # returns nothing because we drop nan values
data["Generosity"].fillna('empty',inplace = True)
assert  data['Generosity'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example
#assert data.columns[1] == 'Country or region'
#assert data.Score.dtypes == np.float64
