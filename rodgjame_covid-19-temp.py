# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.
# Get the raw data

full_table = pd.read_csv('/kaggle/input/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')

full_table.head()
# Clean up the data



# drop colums we're not going to use

full_table = full_table.drop(columns=['Province/State', 'Lat', 'Long'])



# Group by country and get the sum of cases

full_table = full_table.groupby(['Country/Region']).sum()



# Remove rows that only have 0 for values

full_table = full_table[(full_table.T != 0).any()]



# Sort by most cases today

last_colum_index = len(full_table.columns.values) - 1

last_colum = full_table.columns.values[last_colum_index]

full_table = full_table.sort_values(by=last_colum, ascending=False)



# reduce to top 10

full_table = full_table.head(10)
from IPython.core.pylabtools import figsize

figsize(25,10)



data_hash = {}

for country in full_table.index:

    data_hash[country] = full_table.loc[country].values.tolist()



df = pd.DataFrame(data_hash, index=full_table.columns.values.tolist())

lines = df.plot.line()