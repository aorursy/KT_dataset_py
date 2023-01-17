import numpy as np

import pandas as pd
suicide_data = pd.read_csv('../input/master.csv')
suicide_data.head()
suicide_data.columns.values
suicide_data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicidesper100kpop', 'country-year', 'HDI for year',

       'gdp_for_year_dollars', 'gdp_per_capita_dollars', 'generation']

suicide_data.columns.values
#gdp_for_year is numerical feature, but due to comma seperated number it is stored as string

suicide_data['gdp_for_year_dollars'] = suicide_data['gdp_for_year_dollars'].str.replace(',','').astype(np.int64)

suicide_data.info()
suicide_data.isnull().sum()
suicide_newdata = suicide_data.drop(['HDI for year', 'country-year'],axis=1)

suicide_newdata.describe()
suicide_newdata.describe(include='object')
#Visualizing the dataset

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')
suicide_newdata[['sex','suicides_no']].groupby('sex').mean().plot(kind='bar')
plt.figure(figsize=(14,10))

sns.barplot(x='age',y='suicides_no',data=suicide_newdata.groupby(['age','sex']).sum().reset_index(),hue='sex')
country_suicides = suicide_newdata[['country','suicides_no']].groupby(['country']).sum()
country_suicides.plot(kind='bar', figsize=(40,10), fontsize=25)

country_suicides = country_suicides.reset_index().sort_values(by='suicides_no',ascending =False)
Top_10 = country_suicides[:10]

sns.barplot(x='country', y='suicides_no', data=Top_10).set_title('countries with most suicides')

plt.xticks(rotation=90)
bottom_10 = country_suicides[-10:]

sns.barplot(x='country', y='suicides_no', data=bottom_10).set_title('countries with least suicides')

plt.xticks(rotation=90)

suicide_newdata[['year','suicides_no']].groupby(['year']).sum().plot()
sns.countplot(x='generation',data=suicide_newdata)

plt.xticks(rotation=45)

gen_year = suicide_newdata[['suicides_no','generation','year']].groupby(['generation','year']).sum().reset_index()
plt.figure(figsize=(30,15))

sns.set(font_scale=1.5)

plt.xticks(rotation=90)

sns.barplot(y='suicides_no', x='year',

            hue='generation', data=gen_year, palette='deep').set_title('Suicides vs generations per year')
sns.heatmap(data= suicide_newdata.corr(),annot=True, fmt='.1f',cmap='mako',linewidths=.5)
plt.figure(figsize=(25,15))

plt.xticks(rotation=90)

sns.barplot(x='country', y='population', hue='sex', data=suicide_newdata).set_title('country vs population')