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




# We do not need the first 3 rows (indexes 0 - 2) in our dataset

# Also the column "Unnamed: 0" can be renamed to more appropriately to "Country"

pd.set_option('mode.chained_assignment', None) #to suppress copy reference update warning

df = pd.read_csv('/kaggle/input/obesity-among-adults-by-country-19752016/data.csv')

data = df[3:]

pd.set_option('mode.chained_assignment', None)

data.reset_index(drop = True, inplace = True)

data.rename(columns = {'Unnamed: 0' : 'Country'}, inplace = True)

data.head()
# The following section is for rearranging the data 

# Step 1: Unpivot the current layout

# Step 2: We are interested in the % obesity values (don't know the interpretation of the range in square bracket)

# Step 3: Sort the data and reset the index values

# Step 4: Notice that "Year" indicates overall %, "Year.1" indicates male % and "Year.2" indicates female %

# Step 5: Expand the frame to include the Category as a column

# Step 6: At this point perform a sorting and look for any "No data" entries. Delete those rows

data_rearranged = data.melt(id_vars = ['Country'], var_name = 'Year', value_name = 'Obesity')

data_rearranged['Obesity'] = data_rearranged['Obesity'].apply(lambda x : x.split()[0])

data_rearranged = data_rearranged.sort_values(by = ['Country', 'Year'])

data_rearranged = data_rearranged.reset_index(drop = True)

data_rearranged[['Year', 'Category']] = data_rearranged['Year'].str.split('.', expand = True)

data_rearranged['Category'] = data_rearranged['Category'].map({None: 'Overall', '1': 'Male', '2': 'Female'})

data_rearranged = data_rearranged.set_index(['Country'])

data_rearranged = data_rearranged.drop(['Monaco', 'San Marino', 'Sudan', 'South Sudan'], axis = 0)

data_rearranged = data_rearranged.reset_index(drop = False)



# At this point we have a cleaned dataset

data_rearranged.head()
# Here's a function that will return top obese countries given number of countries and year

def get_topn_countries(dframe, n, year):

    dframe["Obesity"] = dframe["Obesity"].apply(lambda x: float(x))

    aggregated_data = dframe[(dframe["Year"] == year) & (dframe["Category"] == "Overall")].groupby("Country").agg({'Obesity' : 'sum'}).sort_values(by = "Obesity", ascending = False)

    topn_countries = aggregated_data.head(n)

    return topn_countries
import matplotlib.pyplot as plt

def display_obesity_plot(dframe):

    dframe.plot(kind = 'bar', color = 'y', title = 'Obesity Chart', fontsize = 10)

    plt.ylabel('Obese % of population')
display_obesity_plot(get_topn_countries(data_rearranged, 10, '2008'))