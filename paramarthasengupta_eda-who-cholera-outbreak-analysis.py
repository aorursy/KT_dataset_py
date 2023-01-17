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
file=pd.read_csv('/kaggle/input/cholera-dataset/data.csv')

file.head()
file.dtypes
file.isnull().sum()
file.replace(np.nan,0,regex=True,inplace=True)
file.isnull().sum()
file[(file['Number of reported deaths from cholera']=='Unknown') | (file['Number of reported cases of cholera']=='Unknown')]
file.replace('Unknown',0,regex=True,inplace=True)
country_list=file.Country.unique()

len(file.Country.unique())
file[file['Number of reported cases of cholera']=="3 5"]
file['Number of reported cases of cholera'] = file['Number of reported cases of cholera'].str.replace('3 5','0')

file['Number of reported deaths from cholera'] = file['Number of reported deaths from cholera'].str.replace('0 0','0')

file['Cholera case fatality rate'] = file['Cholera case fatality rate'].str.replace('0.0 0.0','0')
import seaborn as sns

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(30,90))

for c,num in zip(file.Country.unique(), np.arange(1,len(file.Country.unique()))):

    file_req=file[file['Country']==c]

    ax = fig.add_subplot(27,6,num)

    x=file_req['Year']

    y1=pd.to_numeric(file_req['Number of reported cases of cholera'])

    ax.plot(x,y1)

    y2=pd.to_numeric(file_req['Number of reported deaths from cholera'])

    ax.plot(x,y2,color='r')

    ax.legend(['Reported Cases','Deaths'])

    ax.set_title(c)

    ax.set_xlabel('Years')

    ax.set_ylabel('Cases Count')



plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(30,10))

for c,num in zip(file['WHO Region'].unique(), np.arange(1,1+len(file['WHO Region'].unique()))):

    file_req=file[file['WHO Region']==c]

    ax = fig.add_subplot(2,3,num)

    x=file_req['Year'].unique()

    y11=file_req.groupby('Year').apply(lambda x:np.sum(pd.to_numeric(x['Number of reported cases of cholera']))).reset_index(name='Counts')

    y1=y11['Counts']

    sns.regplot(x,y1)

    y21=file_req.groupby('Year').apply(lambda x:np.sum(pd.to_numeric(x['Number of reported deaths from cholera']))).reset_index(name='Counts')

    y2=y21['Counts']

    sns.regplot(x,y2)

    ax.legend(['Reported Cases','Deaths'])

    ax.set_title(c)

    ax.set_xlabel('Years')

    ax.set_ylabel('Cases Count')



plt.tight_layout()

plt.show()
file_ctry=file.groupby(['Country']).apply(lambda x:np.sum(pd.to_numeric(x['Number of reported cases of cholera']))).reset_index(name='Count of Cholera Cases')

print('The Number of Cholera Cases for the counties:\n',file_ctry)

plt.figure(figsize=(30,10))

plt.scatter(file_ctry['Country'],file_ctry['Count of Cholera Cases'])

plt.xlabel('Countries',size=20)

plt.ylabel('Cholera Cases Count',size=20)

plt.title('Cholera Cases Distribution- All Time',size=40)

plt.xticks(rotation=90)

plt.show()
import geopandas as gpd

fp = "/kaggle/input/world-countries-shp-file/TM_WORLD_BORDERS-0.3.shp"

map_df = gpd.read_file(fp)
map_df.name.replace({'Burma':'Myanmar'},regex=True,inplace=True)

map_df.name.replace({'Korea, Republic of':'Republic of Korea'},regex=True,inplace=True)

map_df.name.replace({'Russia':'Russian Federation'},regex=True,inplace=True)

map_df.name.replace({'United Kingdom':'United Kingdom of Great Britain and Northern Ireland'},regex=True,inplace=True)

map_df.name.replace({'United States':'United States of America'},regex=True,inplace=True)

map_df.name.replace({ 'Venezuela': 'Venezuela (Bolivarian Republic of)'},regex=True,inplace=True)
merged = map_df.set_index('name').join(file_ctry.set_index('Country'))

variable = 'Count of Cholera Cases'

vmin, vmax = np.min(merged['Count of Cholera Cases']), np.max(merged['Count of Cholera Cases'])

fig, ax = plt.subplots(1, figsize=(20, 12))

merged.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')

ax.axis('off')

ax.set_title('Cholera Cases History: World Level', fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate('Source:Cholera Dataset',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)
file_yrs=file.Year.unique()

file_yrs.sort()

final_10_yrs=file_yrs[-10:]

final_10_yrs
ten_yr_case=file[file['Year'].isin(final_10_yrs)]

file_yr_ctry=ten_yr_case.groupby(['Country']).apply(lambda x:np.sum(pd.to_numeric(x['Number of reported cases of cholera']))).reset_index(name='Count of Cholera Cases')

best_cases_country=file_yr_ctry.sort_values(by='Count of Cholera Cases',ascending=True)[0:10]

print('Countries having the least number of Cholera Cases in the last 10 years: \n',best_cases_country)
merged = map_df.set_index('name').join(best_cases_country.set_index('Country'))

variable = 'Count of Cholera Cases'

vmin, vmax = np.min(merged['Count of Cholera Cases']), np.max(merged['Count of Cholera Cases'])

fig, ax = plt.subplots(1, figsize=(20, 12))

merged.plot(column=variable, cmap='Greens', linewidth=0.8, ax=ax, edgecolor='0.8')

ax.axis('off')

ax.set_title('Countries with the Least Count of Cholera (Last 10 Years)', fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate('Source:Cholera Dataset',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

sm = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)
worst_cases_country=file_yr_ctry.sort_values(by='Count of Cholera Cases',ascending=False)[0:10]

print('Countries having the most number of Cholera Cases in the last 10 years: \n',worst_cases_country)
merged = map_df.set_index('name').join(worst_cases_country.set_index('Country'))

variable = 'Count of Cholera Cases'

vmin, vmax = np.min(merged['Count of Cholera Cases']), np.max(merged['Count of Cholera Cases'])

fig, ax = plt.subplots(1, figsize=(20, 12))

merged.plot(column=variable, cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8')

ax.axis('off')

ax.set_title('Countries with the Highest Count of Cholera (Last 10 Years)', fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate('Source:Cholera Dataset',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)
file_ctry_deaths=file.groupby(['Country']).apply(lambda x:(pd.to_numeric(x['Number of reported deaths from cholera'])).sum()).reset_index(name='Count of Deaths from Cholera')

print('The Number of Cholera Cases for the counties:\n',file_ctry_deaths)

plt.figure(figsize=(30,10))

plt.bar(file_ctry_deaths['Country'],file_ctry_deaths['Count of Deaths from Cholera'])

plt.xlabel('Countries',size=30)

plt.ylabel('Cholera Death Cases Count',size=30)

plt.title('Cholera Death Cases Distribution- All Time',size=40)

plt.xticks(rotation=90)

plt.show()

merged = map_df.set_index('name').join(file_ctry_deaths.set_index('Country'))

variable = 'Count of Deaths from Cholera'

vmin, vmax = np.min(merged['Count of Deaths from Cholera']), np.max(merged['Count of Deaths from Cholera'])

fig, ax = plt.subplots(1, figsize=(20, 12))

merged.plot(column=variable, cmap='PuBu', linewidth=0.8, ax=ax, edgecolor='0.8')

ax.axis('off')

ax.set_title('Countries with the Highest Count of Death from Cholera', fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate('Source:Cholera Dataset',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

sm = plt.cm.ScalarMappable(cmap='PuBu', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)
file_yr_region=ten_yr_case.groupby(['WHO Region','Year']).apply(lambda x:np.sum(pd.to_numeric(x['Number of reported cases of cholera']))).reset_index(name='Count of Cholera Cases')

heatmap1_data = pd.pivot_table(file_yr_region, values='Count of Cholera Cases', 

                     index=['WHO Region'],columns='Year')

plt.figure(figsize=(10,10))

sns.heatmap(heatmap1_data,annot=True,fmt='.0f', cmap='YlOrBr').set_title('WHO Region vs Year Heatmap of Cholera Cases- in the last 10 years',size=15)
fat_ctry=file.groupby('Country').apply(lambda x:pd.to_numeric(x['Cholera case fatality rate']).median()).reset_index(name='Median Fatality')

fat_ctry.replace(np.NaN,0,inplace=True)

plt.figure(figsize=(30,5))

plt.plot(fat_ctry['Country'],fat_ctry['Median Fatality'],color='r')

plt.xlabel('Country',size=25)

plt.ylabel('Fatality Rate (Median)',size=20)

plt.title('Median Fatality Rate- Nation wise',size=40)

plt.xticks(rotation=90)

merged = map_df.set_index('name').join(fat_ctry.set_index('Country'))

variable = 'Median Fatality'

vmin, vmax = np.min(merged['Median Fatality']), np.max(merged['Median Fatality'])

fig, ax = plt.subplots(1, figsize=(20, 12))

merged.plot(column=variable, cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8')

ax.axis('off')

ax.set_title('Countries with the Highest Fatality Rates from Cholera', fontdict={'fontsize': '25', 'fontweight' : '3'})

ax.annotate('Source:Cholera Dataset',xy=(0.1, .08),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')

sm = plt.cm.ScalarMappable(cmap='PuBuGn', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)
def country_summary(country):

    country_file=file[file['Country']==country]

    country_case_counts=country_file.groupby('Year').apply(lambda x:pd.to_numeric(x['Number of reported cases of cholera'])).reset_index(name='Total Cases of Cholera')

    country_death_counts=country_file.groupby('Year').apply(lambda x:pd.to_numeric(x['Number of reported deaths from cholera'])).reset_index(name='Total Deaths from Cholera')

    plt.plot(country_case_counts['Year'],country_case_counts['Total Cases of Cholera'])

    plt.plot(country_death_counts['Year'],country_death_counts['Total Deaths from Cholera'],color='r')

    plt.xlabel('Year')

    plt.ylabel('Total count numbers')

    plt.title('Analysis of Cholera Cases vs Death: {}'.format(country))

    plt.legend(['Total Cases of Cholera','Total Deaths from Cholera'])

    print('The Year {:.0f} has seen the maximum number of Cholera Cases: {:.0f}'.format(country_case_counts.loc[np.max(country_case_counts['Total Cases of Cholera'].idxmax())]['Year'],np.max(country_case_counts['Total Cases of Cholera'])))

    print('The Year {:.0f} has seen the maximum number of deaths due to Cholera: {:.0f}'.format(round(country_death_counts.loc[np.max(country_death_counts['Total Deaths from Cholera'].idxmax())]['Year']),round(np.max(country_death_counts['Total Deaths from Cholera']))))
India_Data=country_summary('India')
Bangladesh_Data=country_summary('Bangladesh')
Haiti_Data=country_summary('Haiti')