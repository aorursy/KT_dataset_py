# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import plotly

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv')
df_pop = pd.read_csv('../input/populations-by-state/populations.csv')
df.head()
df.info()
df.date = pd.to_datetime(df.date)
df.n_killed.mean()
df.sort_values(by='n_killed', ascending=False).head()
df.groupby('state').n_killed.sum().sort_values(ascending=False)
ax = df.groupby('state').n_killed.sum().sort_values().plot(kind='barh', figsize=(10,10));
ax.set_title('Deaths by State');
ax.set_xlabel('# of Deaths');
df['Year'] = df.date.dt.year

df.Year.value_counts().sort_index().plot();
plt.title('Incidents over Time');
plt.ylabel('Incidents');
plt.xlabel('Year');
# months = df.date.dt.month
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111)
# bins=[1,2,3,4,5,6,7,8,9,10,11,12]
# ax.hist(months, bins, histtype='bar', alpha=0.7, color='orange', rwidth=0.8, ec='black', range=(1,12));
# #ax.set_xlim(left=1,right=12);
# ax.set(title='Incidents and Months',xlabel='months',ylabel='count #');


ax = df.date.dt.month.value_counts().sort_index().plot('bar',figsize=(8,8),title='Incidents in Months',x='Count #',y='Months',
                                                       ec='black', color='orange', alpha=0.8);
ax.set_xlabel('Months');
ax.set_ylabel('Count #');




# create new gun dataframe
# remove unknown from guntype

guns = df[(df.gun_type.str.contains('\|') == False) & (df.gun_type.str.contains('Unknown') == False)]

#look at different guns
top_guns = guns.gun_type.value_counts().sort_values(ascending=False).head(20)

#take colons out of strings
top_guns.index = top_guns.index.str.replace(':','')

#remove 0 from beginning of name of gun
top_guns.index = top_guns.index.str.replace('^0','')


# create plot for guns
plt.figure(figsize=(10,10));
plt.barh(y=top_guns.index,width=top_guns, ec='black', alpha=0.8, align='center');
plt.xlabel('Incident Count');
plt.ylabel('Gun Type');
plt.title('# of Guns in Incidents');

# remove unneeded columns
df_pop.drop(['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5','Unnamed: 6', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'], axis=1, inplace=True)
# remove unneeded rows
df_pop = df_pop.iloc[8:59]
# rename columns
df_pop.columns = ['State', 'Population']

# remove '.' from beginning of State column
df_pop.State = df_pop.State.str.replace('.','')

df.groupby('state').n_killed.sum() 
df_pop.Population = df_pop.Population.str.replace(',','').astype('int64')

# merge both dataframe into a new dataframe

d = df.groupby('state').n_killed.sum() 

#create new dataframe with state, population, and proportions (killed per capita)
df1 = pd.DataFrame(data={'state':df_pop.State,'n_killed': d.values, 'population': df_pop.Population})

df1['proportion'] = df1.n_killed / df1.population

# sort dataframe by proportion
df1.sort_values(by='proportion', inplace=True)

plt.figure(figsize=(10,10));
plt.barh(width = df1.proportion, y=df1.state, color='orange', alpha=0.7, ec='black', lw=0.8);
plt.title('Number of deaths per Capita');
plt.xlabel('Porportion of deaths per state');
plt.ylabel('State');

