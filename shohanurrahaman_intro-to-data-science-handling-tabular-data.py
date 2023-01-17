# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

display(df.head())
#select individual columns 

pop = df['population']

display(pop)



#select multiple columns 

cols_to_select = ['population','total_rooms', 'total_bedrooms']

cols = df[cols_to_select]

display(cols)
#inplace doesn't copy a new dataframe 

df.rename(columns = {'population':'pop'}, inplace=True)

display(df.head(2))
unique = df['ocean_proximity'].unique()

print(unique)
island_cond = df['ocean_proximity'] == 'ISLAND'

island_data = df[island_cond]



print('shape of original data :', df.shape)

print('shape of filtered data : ', island_data.shape)



#have only 5 columns those ocean_proximity is ISLAND

display(island_data.head(10))
print('shape of original data :', df.shape)



#population less than 10k

pop_cond = df['pop'] < 10000 

pop_data = df[pop_cond]

print(pop_data.shape)



#population greather than 15000 and households less than 5000

pop_ho = (df['pop'] > 15000) & (df['households'] < 5000)

pod_ho_data = df[pop_ho]

print(pod_ho_data.shape)



#house near ocean or also near bay

cond = ['NEAR OCEAN', 'NEAR BAY']

near_ocean_bay = df['ocean_proximity'].isin(cond)

nead_ocean_bay_df = df[near_ocean_bay]

print(nead_ocean_bay_df.shape)
