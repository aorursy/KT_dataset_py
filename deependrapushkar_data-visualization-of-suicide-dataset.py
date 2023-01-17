import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import os

print(os.listdir("../input"))   
data = pd.read_csv('../input/master.csv')

data.head()
data.tail()
data.shape
data.describe()
data.info()
data.isnull().any()
data['HDI for year'].isna().sum()
fig = plt.figure(figsize=(20,2))

sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = 'ocean')
data = data.drop('HDI for year',axis =1)
data.columns
data[' gdp_for_year ($) '] = data[' gdp_for_year ($) '].apply(lambda x: x.replace(',','')).astype(float)
corr = data.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize=(15,10))

sns.heatmap(corr, annot = True,mask = mask, linewidth = 0.3)

plt.title('Suicide data correlation')

plt.show()
# now we can see the pattern of suicides in respect to population of countries

plt.figure(figsize=(20,10))

p= sns.barplot(x="country",y = 'suicides/100k pop', data=data, palette='colorblind')

p.set_title("variation  of suicide per 100k population with countries")

for i in p.get_xticklabels():

    i.set_rotation(90)
#now let us see how suicide rates according to gender are distributed



gender_data = data.groupby(['sex'])['suicides_no'].sum().reset_index()

sns.barplot(x ='sex',y='suicides_no',data = gender_data)

plt.show()
plt.figure(figsize=(15,10))

age_data = data.groupby(['age'])['suicides_no'].sum().reset_index()

sns.barplot(x ='age',y='suicides_no',data = age_data)

plt.show()
plt.figure(figsize=(10,6))

gen_data = data.groupby(['generation'])['suicides_no'].sum().reset_index()

gen_data = gen_data.sort_values(by='suicides_no',ascending =False)

sns.barplot(x ='generation',y='suicides_no',data = gen_data)

plt.show()
#distribution of suicide_no in generations in the form of pie plot

plt.figure(figsize=(18,8))

gen =['Boomers','Silent','Generation X','Millenials','G.I. Generation','Generation Z']

plt.pie(data['generation'].value_counts(),explode=[0.1,0.1,0.1,0.1,0.1,0.1],labels =gen, startangle=90, autopct='%.1f%%')

plt.title('Generations Count')

plt.ylabel('Count')
# let's see the countries having highest suicide rates

plt.figure(figsize=(15,6))

con_data = data.groupby(['country'])['suicides_no'].sum().reset_index()

con_data = con_data.sort_values(by='suicides_no',ascending =False)

con_data = con_data.head(10)

sns.barplot(x ='country',y='suicides_no',data = con_data)

plt.show()
# countries with least suicide rates

plt.figure(figsize=(15,6))

con1_data = data.groupby(['country'])['suicides_no'].sum().reset_index()

con1_data = con1_data.sort_values(by='suicides_no',ascending =False)

con1_data = con1_data.tail(10)

sns.barplot(x ='country',y='suicides_no',data = con1_data)

plt.show()
array = ['Russian Federation', 'United States', 'Japan', 'France', 'Ukraine', 'Germany', 'Republic of Korea', 'Brazil', 'Poland', 'United Kingdom']

Period = data.loc[data['country'].isin(array)]

Period = Period.groupby(['country', 'year'])['suicides_no'].sum().unstack('country').plot(figsize=(20, 7))

Period.set_title('Top suicide countries', size=15, fontweight='bold')
array = ['Russian Federation', 'United States', 'Japan', 'France', 'Ukraine', 'Germany', 'Republic of Korea', 'Brazil', 'Poland', 'United Kingdom']

gdp_Period = data.loc[data['country'].isin(array)]

gdp_Period = gdp_Period.groupby(['country', 'year'])['gdp_per_capita ($)'].sum().unstack('country').plot(figsize=(20, 7))

gdp_Period.set_title('Top per capita gdp countries', size=15, fontweight='bold')
fig=sns.jointplot(y='gdp_per_capita ($)',x='suicides_no',kind='hex',data=data[data['country']=='United States'])

plt.show()
plt.figure(figsize=(15,6))

con2_data = data.groupby(['country'])['population'].sum().reset_index()

con2_data = con2_data.sort_values(by='population',ascending =False)

con2_data = con2_data.head(10)

sns.barplot(x ='country',y='population',data = con2_data)

plt.show()
plt.figure(figsize=(15,6))

sns.barplot(x ='year',y='population',data = (data[data['country'] == 'United States']))

plt.show()
plt.figure(figsize=(15,6))

sns.barplot(x ='year',y='population',data = (data[data['country'] == 'Russian Federation']))

plt.show()
plt.figure(figsize=(15,7))

sns.stripplot(x="year",y='suicides/100k pop',data=data)

plt.xticks(rotation=45)

plt.show()
dfSexPeriod =data.groupby(['sex', 'year'])['suicides_no'].sum().unstack('sex').plot(figsize=(20, 7))

dfSexPeriod.set_title('Suicide per Sex', size=15, fontweight='bold')
dfAgePeriod =data.groupby(['age', 'year'])['suicides_no'].sum().unstack('age').plot(figsize=(20, 10))

dfAgePeriod.set_title('Suicide per Age', size=15, fontweight='bold')