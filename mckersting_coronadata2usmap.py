# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

"""

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

"""

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
usTotal_df = pd.read_csv('/kaggle/input/uncover/UNCOVER/covid_tracking_project/covid-statistics-totals-for-all-of-us.csv')

usTotal_df.head()
positive_df = usTotal_df[['state_name', 'positive', 'total']]

positive_df.drop([3, 12, 27, 50, 56], inplace=True)
import geopandas

country = geopandas.read_file('/kaggle/input/us-states-map/gz_2010_us_040_00_500k.json')
country.head()
country.plot()
#Goal is to get a map like this showing the density of cases

country[country['NAME'].isin(['Alaska','Hawaii']) == False].plot(column='CENSUSAREA',figsize=(30,20))
#Sort the names alphabetically and reset indices to match the other df

country.sort_values(by=['NAME'], inplace=True)

country.reset_index(drop=True, inplace=True)
positive_df.sort_values(by=['state_name'], inplace=True)

positive_df.reset_index(drop=True, inplace=True)
#I am keeping the names to make sure the data sets are aligned correctly

country['names_1'] = positive_df['state_name'] 

country['positive'] = positive_df['positive'] 

country['total'] = positive_df['total'] 
country.head()
country.tail()
ax = country[country['NAME'].isin(['Alaska','Hawaii']) == False].plot(column='total',figsize=(30,20), legend=True)

ax.set_title("Total Number of Covid-19 Cases", fontsize=25)
population_df = pd.read_excel('/kaggle/input/populationdata/nst-est2019-01.xlsx')

population_df.head()
population_df.drop([0, 1, 2, 3, 4, 5, 6, 7, 59, 61, 62, 63, 64, 65], inplace=True)
population_df.reset_index(drop=True, inplace=True)
population_df.rename(columns={"table with row headers in column A and column headers in rows 3 through 4. (leading dots indicate sub-parts)": "name", "Unnamed: 12": "2019 Population"}, inplace=True)

popualtion_df = population_df['name'][51] = '.Puerto Rico'
population_df.sort_values(by=['name'], inplace=True)
population_df.reset_index(drop=True, inplace=True)
country['names_2'] = population_df['name'] 

country['population'] = population_df['2019 Population'] 
country['Infection rate'] = country['total'] / country['population'] * 100
country.head()
country.tail()
ax = country[country['NAME'].isin(['Alaska','Hawaii']) == False].plot(column='Infection rate',figsize=(30,20), legend=True)

ax.set_title("Percent of population with Covid-19", fontsize=25)
#This data set is three months old. Lesson learned