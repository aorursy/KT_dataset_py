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
df = pd.read_csv('/kaggle/input/data-jobs-listings-glassdoor/glassdoor.csv')
df.shape
df.head()
for column in df.columns:

    print(column)
df['map.country'].isnull().sum()

countriesData = df['map.country'].dropna()

countriesData
countriesData.unique()
 #What is great about this function is that it returns always the same output for all inputs corresponding to identifiers of the country

from iso3166 import countries

print(countries.get('us'))

print(countries.get('USA'))

print(countries.get('United States of America'))
def rename(country):

    try:

        return countries.get(country).alpha3

    except:

        return (np.nan)
old_sample_number = countriesData.shape[0]



countriesData = countriesData.apply(rename)

countriesData = countriesData.dropna()



new_sample_number = countriesData.shape[0]

print('we lost', old_sample_number-new_sample_number, 'samples after converting')
countriesData
import matplotlib.pyplot as plt

import seaborn as sns 

plt.figure(figsize=(24, 6))

sns.barplot(countriesData.value_counts()[countriesData.value_counts()>150].index, countriesData.value_counts()[countriesData.value_counts()>150].values)
#Creating a DataFrame that stores the ID of the countries and their count

country_df = pd.DataFrame(data=[countriesData.value_counts().index, countriesData.value_counts().values],index=['country','count']).T
#Converting count values to int because this will be important for plotly

country_df['count']=pd.to_numeric(country_df['count'])
country_df.head()
import plotly.express as px

fig = px.scatter_geo(country_df, locations="country", size='count',

                     hover_name="country", color='country',

                     projection="natural earth")

fig.show()