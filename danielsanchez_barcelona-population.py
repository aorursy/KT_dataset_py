import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
df_population = pd.read_csv('../input/population.csv')
df_births = pd.read_csv('../input/births.csv')
df_deaths = pd.read_csv('../input/deaths.csv')
df_population.head()
df_population.isnull().sum().sum()
population_per_year = df_population.groupby('Year')['Number'].sum().reset_index()
ax = sns.lineplot(x="Year", y="Number", data=population_per_year)
ax.set_xticks(population_per_year.Year)
ax.set_title('Population per year')
ax.set_ylabel('Population')
fig=plt.gcf()
fig.set_size_inches(18,6)
df_population[(df_population.Year == 2014) & (df_population.Gender=='Male')].Number.sum()
df_population.groupby(['Year', 'Gender'])['Number'].sum().unstack().plot.bar(title='Population per year and gender')
fig=plt.gcf()
fig.set_size_inches(18,6)
year = 2017
population_age_gender = df_population[df_population.Year == year].groupby(['Age', 'Gender'])['Number'].sum().unstack().reset_index()
population_age_gender['order'] = [0, 2, 3, 4, 5, 6, 7, 8, 9, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
population_age_gender = population_age_gender.sort_values('order')

women_bins = population_age_gender['Female'].values 
men_bins = population_age_gender['Male'].values * (-1)

ages = population_age_gender['Age'].values

layout = go.Layout(yaxis=go.layout.YAxis(title='Age'),
                   xaxis=go.layout.XAxis(
                       range=[-80000, 80000],
                       tickvals=np.arange(-80000, 80001, 10000),
                       ticktext=np.arange(-80000, 80001, 10000),
                       title='Number'),
                   barmode='overlay',
                   bargap=0.1)

data = [go.Bar(y=ages,
               x=men_bins,
               orientation='h',
               name='Men',
               text=-1 * men_bins.astype('int'),
               hoverinfo='text',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=ages,
               x=women_bins,
               orientation='h',
               name='Women',
               text=women_bins.astype('int'),
               hoverinfo='text',
               marker=dict(color='seagreen')
               )]

py.iplot(dict(data=data, layout=layout), filename='population-pyramid') 
f,ax = plt.subplots(1,2, figsize=(25,9))

district_populations = df_population.groupby('District.Name')['Number'].sum().reset_index().sort_values('Number',ascending=False)
sns.barplot(x='Number', y='District.Name', data=district_populations, ax=ax[0])
ax[0].set_title('Population by District')
ax[0].set_xlabel('Population')

neighborhood_populations = df_population.groupby('Neighborhood.Name')['Number'].sum().reset_index().sort_values('Number',ascending=False).iloc[0:10]
sns.barplot(x='Number', y='Neighborhood.Name', data=neighborhood_populations, ax=ax[1])
ax[1].set_title('Population by Neighborhood (Top 10 most populated)')
ax[1].set_xlabel('Population')

plt.subplots_adjust(hspace=0.3,wspace=0.6)
ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)
plt.show()
f,ax = plt.subplots(1,1, figsize=(20,7))

neighborhood_populations = df_population.groupby('Neighborhood.Name')['Number'].sum().reset_index().sort_values('Number',ascending=True).iloc[0:10]
sns.barplot(x='Number', y='Neighborhood.Name', data=neighborhood_populations, ax=ax)
ax.set_title('Population by Neighborhood (Top 10 less populated)')
ax.set_xlabel('Population')
ax.tick_params(labelsize=15)
plt.show()
# population_discrict_year = df_population.groupby(['District.Name', 'Year'])['Number'].sum().reset_index()
population_discrict_year = df_population.groupby(['District.Name', 'Year'])['Number'].sum().reset_index()
districts = district_populations.sort_values('Number', ascending=False)['District.Name'].values[0:10]

diff_population_district_2016 = []
diff_population_district_2013 = []
for district in districts:
    df_district = population_discrict_year[population_discrict_year['District.Name'] == district].sort_values('Year', ascending=True)
    diff_population_district_2016.append(df_district[df_district['Year'] == 2017]['Number'].values[0] - df_district[df_district['Year'] == 2016]['Number'].values[0])
    diff_population_district_2013.append(df_district[df_district['Year'] == 2017]['Number'].values[0] - df_district[df_district['Year'] == 2013]['Number'].values[0])
    
df_district_growth = pd.DataFrame()
df_district_growth['District'] = districts
df_district_growth['Population Growth from 2016'] = diff_population_district_2016
df_district_growth['Population Growth from 2013'] = diff_population_district_2013
df_district_growth = df_district_growth.sort_values('Population Growth from 2016', ascending=False)


f,ax = plt.subplots(1,2, figsize=(25,9))

sns.barplot(x='Population Growth from 2016', y='District', data=df_district_growth, ax=ax[0])
ax[0].set_title('Population Growth by District from 2016')
#ax[0].set_xlabel('Population')

df_district_growth = df_district_growth.sort_values('Population Growth from 2013', ascending=False)
sns.barplot(x='Population Growth from 2013', y='District', data=df_district_growth, ax=ax[1])
ax[1].set_title('Population Growth by District from 2013')
#ax[0].set_xlabel('Population')

plt.subplots_adjust(hspace=0.2,wspace=0.2)
ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)
plt.show()
def plot_population_grow(column='District.Name', topk=10, sort_ascending=False, wspace=0.2):
    population_year = df_population.groupby([column, 'Year'])['Number'].sum().reset_index()
    locations = population_year[column].unique()
    
    diff_population_2016 = []
    diff_population_2013 = []
    
    for location in locations:
        df_location = population_year[population_year[column] == location].sort_values('Year', ascending=True)
        diff_population_2016.append(df_location[df_location['Year'] == 2017]['Number'].values[0] - df_location[df_location['Year'] == 2016]['Number'].values[0])
        diff_population_2013.append(df_location[df_location['Year'] == 2017]['Number'].values[0] - df_location[df_location['Year'] == 2013]['Number'].values[0])   
    
    
    df_growth = pd.DataFrame()
    df_growth[column] = locations
    df_growth['Population Growth since 2016'] = diff_population_2016
    df_growth['Population Growth since 2013'] = diff_population_2013
    df_growth = df_growth.sort_values('Population Growth since 2016', ascending=sort_ascending)

    f,ax = plt.subplots(1,2, figsize=(25,9))

    sns.barplot(x='Population Growth since 2016', y=column, data=df_growth.head(topk), ax=ax[0])
    ax[0].set_title('Population Growth by ' + column + ' since 2016')
    #ax[0].set_xlabel('Population')

    df_growth = df_growth.sort_values('Population Growth since 2013', ascending=sort_ascending)
    sns.barplot(x='Population Growth since 2013', y=column, data=df_growth.head(topk), ax=ax[1])
    ax[1].set_title('Population Growth by ' + column + ' since 2013')
    #ax[0].set_xlabel('Population')

    plt.subplots_adjust(hspace=0.2,wspace=wspace)
    ax[0].tick_params(labelsize=15)
    ax[1].tick_params(labelsize=15)
    plt.show()
plot_population_grow(column='District.Name')
plot_population_grow(column='Neighborhood.Name', sort_ascending=False, wspace=0.6)
plot_population_grow(column='Neighborhood.Name', sort_ascending=True)
grouped_death = df_deaths.groupby('District.Name')['Number'].sum().to_frame().reset_index()
grouped_births = df_births.groupby('District Name')['Number'].sum().to_frame().reset_index()
grouped_death_births = grouped_death.merge(grouped_births, how='left', left_on='District.Name', right_on='District Name', suffixes=('_death', '_birth'))
grouped_death_births['natural_growth'] = grouped_death_births['Number_birth'] - grouped_death_births['Number_death']

f,ax = plt.subplots(1,1, figsize=(15,5))

sns.barplot(x='District.Name', y='natural_growth', data=grouped_death_births.sort_values('natural_growth'), ax=ax)
ax.set_title('Natural growth per discrict since 2013')

plt.show()
grouped_death = df_deaths.groupby('Neighborhood.Name')['Number'].sum().to_frame().reset_index()
grouped_births = df_births.groupby('Neighborhood Name')['Number'].sum().to_frame().reset_index()
grouped_death_births = grouped_death.merge(grouped_births, how='left', left_on='Neighborhood.Name', right_on='Neighborhood Name', suffixes=('_death', '_birth'))
grouped_death_births['natural_growth'] = grouped_death_births['Number_birth'] - grouped_death_births['Number_death']

f,ax = plt.subplots(1,2, figsize=(18,6))

sns.barplot(x='natural_growth', y='Neighborhood.Name', data=grouped_death_births.sort_values('natural_growth', ascending=False).head(10), ax=ax[0])
ax[0].set_title('Natural growth per Neighborhood since 2013')

sns.barplot(x='natural_growth', y='Neighborhood.Name', data=grouped_death_births.sort_values('natural_growth', ascending=True).head(10), ax=ax[1])
ax[1].set_title('Natural growth per Neighborhood since 2013')

plt.subplots_adjust(hspace=0.1,wspace=0.3)
ax[0].tick_params(labelsize=10)
ax[1].tick_params(labelsize=10)
plt.show()
year = 2017
population_age_gender = df_population[(df_population.Year == year) & (df_population['Neighborhood.Name'] == 'el Poble Sec')].groupby(['Age', 'Gender'])['Number'].sum().unstack().reset_index()
population_age_gender['order'] = [0, 2, 3, 4, 5, 6, 7, 8, 9, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
population_age_gender = population_age_gender.sort_values('order')

women_bins = population_age_gender['Female'].values 
men_bins = population_age_gender['Male'].values * (-1)

ages = population_age_gender['Age'].values

layout = go.Layout(yaxis=go.layout.YAxis(title='Age'),
                   xaxis=go.layout.XAxis(
                       range=[-4000, 4000],
                       tickvals=np.arange(-4000, 4001, 4000),
                       ticktext=np.arange(-4000, 4001, 4000),
                       title='Population in el Poble Sec'),
                   barmode='overlay',
                   bargap=0.1)

data = [go.Bar(y=ages,
               x=men_bins,
               orientation='h',
               name='Men',
               text=-1 * men_bins.astype('int'),
               hoverinfo='text',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=ages,
               x=women_bins,
               orientation='h',
               name='Women',
               text=women_bins.astype('int'),
               hoverinfo='text',
               marker=dict(color='seagreen')
               )]

py.iplot(dict(data=data, layout=layout), filename='population-pyramid') 
year = 2017
population_age_gender = df_population[(df_population.Year == year) & (df_population['Neighborhood.Name'] == 'el Raval')].groupby(['Age', 'Gender'])['Number'].sum().unstack().reset_index()
population_age_gender['order'] = [0, 2, 3, 4, 5, 6, 7, 8, 9, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
population_age_gender = population_age_gender.sort_values('order')

women_bins = population_age_gender['Female'].values 
men_bins = population_age_gender['Male'].values * (-1)

ages = population_age_gender['Age'].values

layout = go.Layout(yaxis=go.layout.YAxis(title='Age'),
                   xaxis=go.layout.XAxis(
                       range=[-4000, 4000],
                       tickvals=np.arange(-4000, 4001, 4000),
                       ticktext=np.arange(-4000, 4001, 4000),
                       title='Population in el Raval'),
                   barmode='overlay',
                   bargap=0.1)

data = [go.Bar(y=ages,
               x=men_bins,
               orientation='h',
               name='Men',
               text=-1 * men_bins.astype('int'),
               hoverinfo='text',
               marker=dict(color='powderblue')
               ),
        go.Bar(y=ages,
               x=women_bins,
               orientation='h',
               name='Women',
               text=women_bins.astype('int'),
               hoverinfo='text',
               marker=dict(color='seagreen')
               )]

py.iplot(dict(data=data, layout=layout), filename='population-pyramid') 