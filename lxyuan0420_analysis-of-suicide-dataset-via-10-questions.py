import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/master.csv')

data.head()
print ("Data shape:", data.shape, "\n")

print (data.info())
county = data[['country','suicides/100k pop']].groupby('country',as_index=False).mean().sort_values(by='suicides/100k pop',ascending=False)

fig=plt.figure(figsize=(40,10))

sns.barplot(x=county['country'],y=county['suicides/100k pop'],data=county)

plt.xticks(rotation=90);
fig=plt.figure(figsize=(3,5));

sns.barplot(x='sex', y='suicides/100k pop',data=data[['sex','suicides/100k pop']].groupby('sex',as_index=False).mean().sort_values(by='suicides/100k pop',ascending=False));
fig=plt.figure(figsize=(8,6));

sns.barplot(x='age',y='suicides/100k pop',data=data[['age','suicides/100k pop']].groupby('age',as_index=False).mean().sort_values(by='suicides/100k pop',ascending=False));
fig=plt.figure(figsize=(8,7));

sns.barplot(x='age', y='suicides/100k pop',hue='sex', data=data[['age','suicides/100k pop','sex']].groupby(['age','sex'],as_index=False).mean().sort_values(by='suicides/100k pop',ascending=False));
#fig=plt.figure(figsize=(15,5));

fig, ax = plt.subplots(figsize=(20, 5))

plt.title('Suicide rate between 1985-2016');

sns.lineplot(x='year', y='suicides/100k pop', data=data[['year','suicides/100k pop']].groupby('year',as_index=False).mean().sort_values(by='year',ascending=True),marker='o',color='BLACK');

ax.set(xticks=data['year'].unique());
GDP_SR_Year = data[['year','suicides/100k pop', 'gdp_per_capita ($)']].groupby('year').mean().reset_index()



# line graph 1

fig, ax = plt.subplots(figsize=(20, 5))

year_SR = ax.plot(GDP_SR_Year['year'],GDP_SR_Year['suicides/100k pop'],marker='o',color='BLACK', label='Suicide rate')



# line graph 2

ax2 = ax.twinx()

year_GDP = ax2.plot(GDP_SR_Year['year'],GDP_SR_Year['gdp_per_capita ($)'],marker='*',color='BLUE', label='GDP') 



# Joining legends.

lns = year_SR + year_GDP

labels = [l.get_label() for l in lns]

ax.legend(lns, labels, loc=2)



# Setting labels

ax.set_ylabel('Suicides per 100k population')

ax2.set_ylabel('GDP per Capita($)')

ax.set_xlabel('Time(Years)')

ax.set(xticks=data['year'].unique());

plt.title('Relationship between GDP and Suicide rate');
data_1995 = data[data.year == 1995]

print ("Size of Year-1995 data: {} with {} unique countries".format(data_1995.shape, len(data_1995.country.unique())))

print ("Missing countries are: \n{}".format((set(data.country) - set(data_1995.country))))
fig = plt.figure(figsize=(30,10))

sns.barplot(x='country', y='suicides/100k pop', data=data_1995[['country', 'suicides/100k pop']].groupby('country', as_index=False).mean().sort_values(by='suicides/100k pop', ascending=False));

plt.xticks(rotation=90);
fig = plt.figure(figsize=(30,10))

sns.barplot(x='country', y='gdp_per_capita ($)', data=data_1995[['country', 'gdp_per_capita ($)']].groupby('country', as_index=False).mean().sort_values(by='gdp_per_capita ($)', ascending=False));

plt.xticks(rotation=90);
data_2016 = data[data.year == 2016]

print ("Size of Year-2016 data: {} with {} unique countries".format(data_2016.shape, len(data_2016.country.unique())))

print ("Missing countries are: \n{}".format((set(data.country) - set(data_2016.country))))
fig = plt.figure(figsize=(8,5))

plt.subplot(2,1,1)

sns.barplot(x='country', y='suicides/100k pop', data=data_2016[['country', 'suicides/100k pop']].groupby('country', as_index=False).mean().sort_values(by='suicides/100k pop', ascending=False));

plt.xticks(rotation=90);

plt.title('Suicide rate of 2016');



fig = plt.figure(figsize=(8,5))

plt.subplot(2,1,2)

sns.barplot(x='country', y='gdp_per_capita ($)', data=data_2016[['country', 'gdp_per_capita ($)']].groupby('country', as_index=False).mean().sort_values(by='gdp_per_capita ($)', ascending=False));

plt.xticks(rotation=90);

plt.title('GDP of 2016');
year_country = data[['year','country']].groupby('year',as_index=True).nunique()

year_country['year']= year_country.index

fig = plt.figure(figsize=(15,3));

sns.barplot(x='year', y='country', data=year_country,color="salmon");

plt.title('Number of Countries in the Suicide Dataset ');