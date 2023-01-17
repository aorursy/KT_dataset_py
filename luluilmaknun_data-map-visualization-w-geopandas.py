import pandas as pd

import numpy as np

import geopandas as gpd

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt
sns.set_style('darkgrid')

data = pd.read_csv('../input/master.csv')

data.shape
data.info()
data = data.drop('country-year', 1)

df = data.drop('HDI for year', 1)

df.head()
list(df.columns.values)
## map plot based on country map

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

country_data = list(df['country'].unique())

country_geo = list(world['name'])



country_diff = [country for country in country_data if country not in country_geo]

country_diff
# Some country dont have same name country as listed on geomaps

temp = pd.DataFrame(df['country'].replace({'Russian Federation' : 'Russia', 'Republic of Korea': 'Korea',

                                            'Czech Republic' : 'Czech Rep.', 'Bosnia and Herzegovina' : 'Bosnia and Herz.',

                                           'Dominica' : 'Dominican Rep.'}))



df['country'] = temp

country_data = list(df['country'].unique())

country_data
# make a dataframe of suicides_no and country to be plotted

# suicide_sum = pd.DataFrame(df['suicides_no'].groupby(df['country']).sum())

suicide_sum = df.groupby('country', sort=False)["suicides_no"].sum().reset_index(name ='total_suicides')

suicide_sum = suicide_sum.sort_values(by="total_suicides", ascending=False)



suicide_sum.head()
mapped = world.set_index('name').join(suicide_sum.set_index('country')).reset_index()



to_be_mapped = 'total_suicides'

vmin, vmax = 0,1300000

fig, ax = plt.subplots(1, figsize=(25,25))



mapped.dropna().plot(column=to_be_mapped, cmap='Blues', linewidth=0.8, ax=ax, edgecolors='0.8')

ax.set_title('Number of suicides happened in countries', fontdict={'fontsize':30})

ax.set_axis_off()



sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []



cbar = fig.colorbar(sm, orientation='horizontal')
## num of suicides grouped by generations

suicides_gen = df.groupby('generation', sort=False)["suicides_no"].sum().reset_index(name ='total_suicides')

suicides_gen = suicides_gen.sort_values(by='total_suicides', ascending=False)



fig, ax = plt.subplots(1, figsize=(9,5))

sns.barplot(ax=ax,x='total_suicides',y='generation',hue='generation',data=suicides_gen, palette='pastel')

ax.set_title("Number of suicides per generation")
# line plot progress of total suicides number per country per year

# some produce error because plt read the data as scalar, gotta fix it soon

suicides_year_country = df.groupby(['country', 'year'], sort=True)["suicides_no"].sum().reset_index(name ='total_suicides')

suicides_year_country = suicides_year_country.set_index('country')



len_country = len(country_data)

for row in range(len_country//4+1):

    fig, ax = plt.subplots(1,4, figsize=(20,4),sharey=True)

    for column in range(4):

        try:

            current_ax = ax[column]

            country = country_data[row*4+column]

            to_be_plotted = suicides_year_country.loc[country]

            sns.lineplot(ax=current_ax, x='year', y='total_suicides', data=to_be_plotted,palette='Blues')

            current_ax.set_title(country)

        except:

            continue



# fig, ax = plt.subplots(len_country//3+1,3, figsize=(20,130),sharey=True)

# for num in range(len_country):

#     current_ax = ax[num//3,num%3]

#     country = country_data[num]

#     try :

#         to_be_plotted = suicides_year_country.loc[country]

#         sns.lineplot(ax=current_ax, x='year', y='total_suicides', data=to_be_plotted,palette='Blues')

#         current_ax.set_title(country, loc='left')

#     except ValueError:

#         continue