import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



print(os.listdir('../input/suicide-rates-overview-1985-to-2016'))



# Any results you write to the current directory are saved as output
df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

df.head()
df.info()
df.isnull().any()
df['age'].value_counts()



#age is already age group
df['generation'].unique()

#6 types of generation
df.loc[df['generation'] == 'Millenials','age'].value_counts()

#group per generation
df['country'].unique()
#side note because I saw the philippines

plt.style.use('ggplot')

plt.figure(figsize = (10,6))

df.loc[df['country'] == 'Philippines'].groupby('year')['suicides_no'].sum().plot(kind = 'bar')

plt.title('Yearly suicide count in the Ph')
fig = plt.figure(figsize = (10,25))

ax = fig.add_subplot()

ax = sns.countplot(y = 'country', data = df)

ax.set(title = 'data by country', xlabel = 'count of data')
fig = plt.figure(figsize = (5,5))

ax = fig.add_subplot()

df['sex'].value_counts().plot(kind = 'pie', title = 'suicide by gender', use_index = False, ax = ax)

fig.patch.set_facecolor('white')
plt.figure(figsize=(10,7))

cor = sns.heatmap(data = df.corr(), annot = True, cmap = 'RdBu')
df.head()
plt.style.use('ggplot')

plt.figure(figsize=(10,7))

sns.barplot(data = df, x = 'sex', y = 'suicides_no', ci = False, hue = 'age')

plt.ylabel('suicide count')
plt.style.use('ggplot')

plt.figure(figsize=(10,7))

sns.barplot(data = df, x = 'generation', y = 'suicides_no', ci = False)

plt.ylabel('suicide count')
df.head()
plt.figure(figsize=(10,7))

sns.lineplot(x = 'year', y = 'suicides_no', hue = 'age', data = df)
plt.figure(figsize=(10,7))

sns.lineplot(x = 'year', y = 'suicides_no', hue = 'sex', data = df)
gdp_suicide = df.loc[:, ['suicides_no', 'gdp_per_capita ($)']]

gdp_suicide.head()
gdp_suicide.rename(columns = {'gdp_per_capita ($)' : 'gdp_per_capita'}, inplace = True)

gdp_suicide
gdp_mean = gdp_suicide['gdp_per_capita'].mean()

gdp_std = gdp_suicide['gdp_per_capita'].std()



gdp_mean, gdp_std
no_outliers = gdp_suicide[gdp_suicide['gdp_per_capita'].apply(lambda x: (x-gdp_mean)/gdp_std < 3)]
no_outliers.head()
plt.figure(figsize = (10,7))

sns.scatterplot(x = 'gdp_per_capita', y = 'suicides_no', data = no_outliers)
ph = df[df['country'] == 'Philippines']

ph.head()
plt.figure(figsize = (10,7))

sns.lineplot(x = 'year', y = 'suicides/100k pop', data = ph, hue = 'sex')
plt.figure(figsize = (10,7))

sns.barplot(x = 'sex', y = 'suicides_no', data = ph, hue = 'age', ci = False)
country_df = df.set_index('country')

grouped = country_df.groupby('country').mean()

grouped.head()
plt.figure(figsize = (10,7))



top20 = grouped.sort_values(by = 'suicides/100k pop', ascending = False).loc[:,'suicides/100k pop'].head(20)

top20.plot(kind = 'bar')

ph_suicide100k = grouped.loc['Philippines','suicides/100k pop']

ph_suicide100k = pd.Series(ph_suicide100k, name = 'Philippines', index = ['Philippines'])



plt.figure(figsize = (10,7))

top20.append(ph_suicide100k).plot(kind = 'bar')

plt.title('Top20 countries with highest suicide rate and PH')
ph.columns = ph.columns.str.replace(' ','')

ph['gdp_for_year($)'] = ph['gdp_for_year($)'].str.replace(',','').astype('float')
fig = plt.figure(figsize = (12,8))

ax = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)

ph.groupby('year')['gdp_per_capita($)'].mean().plot(ax = ax)

ph.groupby('year')['gdp_for_year($)'].mean().plot(ax = ax2)

ax.set_title('Philippine GDP per capita')

ax2.set_title('Philippine GDP')

ax.locator_params(axis='x', nbins=15)

ax2.locator_params(axis='x', nbins=15)