# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



import pandas as pd # tabular data processing

import geopandas as gpd # geospacial data processing

import seaborn as sns # easy plotting

sns.set_style("darkgrid")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print("Available files in kaggle/input:")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load suicide data:

suicide_data = pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")
# Display some data

print('There are {} entries in the dataset.'.format(len(suicide_data)))

display(suicide_data.head())
# Explore countries:

countries = suicide_data.country.unique()

print("There are {} different countries in the dataset.".format(len(countries)))
# Show on map

world_data = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

countries_data = world_data.loc[world_data['name'].isin(countries)]

ax = world_data.plot(figsize=(20,20), color='whitesmoke', edgecolor='black', zorder=1)

countries_data.plot(color='lightblue', edgecolor='black', ax=ax)
# 'Bad Names' - later for data cleaning (might be unncessary)

print('Countries in the dataset but arent shown on map:')

country_not_on_map = list(set(countries) - set(world_data['name']))

country_not_on_map.sort() # Sort for ease

print(country_not_on_map)

print('A total of {} countries are in the dataset but arent shon on the map.'.format(len(country_not_on_map)))
# Just in case, let's not trust the naturalearth_lowres dataset to contain every country.

country_not_on_map

print('Countries in world but not in the dataset:')

country_not_in_data = list(set(world_data['name']) - set(countries))

country_not_in_data.sort() # Sort for ease

print(country_not_in_data)

print('A total of {} countries are in the geospatial dataset but arent in the suicide dataset.'.format(len(country_not_in_data)))
# Align mismatched names in the geospatial data:

world_data.loc[world_data['name'] == 'Bosnia and Herz.', 'name'] = 'Bosnia and Herzegovina'

world_data.loc[world_data['name'] == 'Czechia', 'name'] = 'Czech Republic'

world_data.loc[world_data['name'] == 'South Korea', 'name'] = 'Republic of Korea'

world_data.loc[world_data['name'] == 'Russia', 'name'] = 'Russian Federation'

world_data.loc[world_data['name'] == 'United States of America', 'name'] = 'United States'



# Replot map:

countries_data = world_data.loc[world_data['name'].isin(countries)]

ax = world_data.plot(figsize=(20,20), color='whitesmoke', edgecolor='black', zorder=1)

countries_data.plot(color='lightblue', edgecolor='black', ax=ax)
# Explore year feature in the dataset:

sns.distplot(suicide_data.year, bins=range(suicide_data.year.min(), suicide_data.year.max() + 1), kde=False)
# Explore sex:

print('Unique genders in our dataset: ' + str(suicide_data.sex.unique()))

print('Statistics: ')

display(suicide_data.sex.value_counts())
# Explore age:

print('Unique age groups in our dataset: ' + str(suicide_data.age.unique()))

print('Statistics: ')

display(suicide_data.age.value_counts())
# Replace 5-14 by 05-14 in order to create right ordering:

suicide_data['age'].replace(to_replace='5-14 years', value='05-14 years', inplace=True)
# Explore generation:

print('Unique generations in our dataset: ' + str(suicide_data.generation.unique()))
# Sorted generations in our dataset:

sorted_generations = ['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials', 'Generation Z']
chart = sns.catplot(x="generation", kind="count", order=sorted_generations, palette="rocket", data=suicide_data)

chart.set_xticklabels(

    rotation=90, 

    fontweight='light',

    fontsize='large')
# Does suicide rates are growing each year?

# Reaarrange dataframe using aggregate:

suicide_data_year_reaarange = suicide_data.groupby('year').agg({'suicides_no' : 'sum', 'population' : 'sum'})

# Use year as regular column (and not index)

suicide_data_year_reaarange = suicide_data_year_reaarange.reset_index()

# Re-create the rate

suicide_data_year_reaarange['suicide_rate'] = suicide_data_year_reaarange['suicides_no'] / suicide_data_year_reaarange['population']

suicide_data_year_reaarange.head()
sns.regplot(x=suicide_data_year_reaarange['year'],

            y=suicide_data_year_reaarange['suicide_rate'])
# Let's do the same but with regard to gender:

# Reaarrange dataframe using aggregate:

suicide_data_year_sex_reaarange = suicide_data.groupby(['year', 'sex']).agg({'suicides_no' : 'sum', 'population' : 'sum'})

# Use year as regular column (and not index)

suicide_data_year_sex_reaarange = suicide_data_year_sex_reaarange.reset_index()

# Re-create the rate

suicide_data_year_sex_reaarange['suicide_rate'] = suicide_data_year_sex_reaarange['suicides_no'] / suicide_data_year_sex_reaarange['population']

suicide_data_year_sex_reaarange.head()
sns.lmplot(x='year',

           y='suicide_rate',

           hue='sex',

           data=suicide_data_year_sex_reaarange)
sns.swarmplot(x=suicide_data_year_sex_reaarange['sex'],

              y=suicide_data_year_sex_reaarange['suicide_rate'])
# TODO: Find a better idea for a plot that focuses solely on males vs females suicide rates.
# Let's do the same but with regard to age:

# Reaarrange dataframe using aggregate:

suicide_data_year_age_reaarange = suicide_data.groupby(['year', 'age']).agg({'suicides_no' : 'sum', 'population' : 'sum'})

# Use year as regular column (and not index)

suicide_data_year_age_reaarange = suicide_data_year_age_reaarange.reset_index()

# Re-create the rate

suicide_data_year_age_reaarange['suicide_rate'] = suicide_data_year_age_reaarange['suicides_no'] / suicide_data_year_age_reaarange['population']

suicide_data_year_age_reaarange.head()
sns.lmplot(x='year',

           y='suicide_rate',

           hue='age',

           data=suicide_data_year_age_reaarange)
sns.swarmplot(x=suicide_data_year_age_reaarange['age'],

              y=suicide_data_year_age_reaarange['suicide_rate'])
# Refer only to age

suicide_data_age_reaarange = suicide_data_year_age_reaarange.groupby('age').agg({'suicides_no' : 'sum', 'population' : 'sum'})

# Use age as regular column (and not index)

suicide_data_age_reaarange = suicide_data_age_reaarange.reset_index()

# Re-create the rate

suicide_data_age_reaarange['suicide_rate'] = suicide_data_age_reaarange['suicides_no'] / suicide_data_age_reaarange['population']

sns.scatterplot(x=suicide_data_age_reaarange['age'],

              y=suicide_data_age_reaarange['suicide_rate'])

# TODO: How to add trend line, as regplot seemingly not working with non-numeric x-axis?

# TODO: Use the same coloring here