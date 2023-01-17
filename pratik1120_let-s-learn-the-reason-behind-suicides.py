import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px



data = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

data['HDI for year'] = data['HDI for year'].fillna(0)
country = data.loc[:,['country','suicides_no']]

country = country.groupby('country')['suicides_no'].sum().reset_index()

country = country.sort_values('suicides_no')

country = country.tail(10)

fig = px.pie(country, names='country', values='suicides_no', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
country = data.loc[:,['country','suicides_no']]

country = country.groupby('country')['suicides_no'].sum().reset_index()

country = country.sort_values('suicides_no')

country = country.head(10)

fig = px.pie(country, names='country', values='suicides_no', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
perc = data.loc[:,["year","country",'suicides_no']]

perc['total_suicides'] = perc.groupby([perc.country,perc.year])['suicides_no'].transform('sum')

perc.drop('suicides_no', axis=1, inplace=True)

perc = perc.drop_duplicates()

perc = perc[(perc['year']>=1990.0) & (perc['year']<=2012.0)]

perc = perc.sort_values("year",ascending = False)



top_countries = ['Russian Federation','United States','Japan','France',"Ukraine"] 

perc = perc.loc[perc['country'].isin(top_countries)]

perc = perc.sort_values("year")

fig=px.bar(perc,x='country', y="total_suicides", animation_frame="year", 

           animation_group="country", color="country", hover_name="country")

fig.show()
sex = data.loc[:,['year','sex','suicides_no']]

sex['total_suicides'] = sex.groupby(['year','sex'])['suicides_no'].transform('sum')

sex.drop('suicides_no', axis=1, inplace=True)

sex = sex.drop_duplicates()

sex = sex[sex['year']>=2000.0]

sex = sex.sort_values("year")

fig=px.bar(sex,x='sex', y="total_suicides", animation_frame="year", 

           animation_group="sex", color="sex", hover_name="sex")

fig.show()
sex = data.loc[:,['year','sex','suicides/100k pop']]

sex['total_suicides'] = sex.groupby(['year','sex'])['suicides/100k pop'].transform('sum')

sex.drop('suicides/100k pop', axis=1, inplace=True)

sex = sex.drop_duplicates()

sex = sex[sex['year']>=2000.0]

sex = sex.sort_values("year")

fig=px.bar(sex,x='sex', y="total_suicides", animation_frame="year", 

           animation_group="sex", color="sex", hover_name="sex")

fig.show()
year = data.loc[:,['year','suicides_no']]

year['total_suicides'] = year.groupby('year')['suicides_no'].transform('sum')

year.drop('suicides_no', axis=1, inplace=True)

year = year.drop_duplicates()

sns.lineplot(data=year, x='year', y='total_suicides')
age = data.loc[:,['year','age','suicides_no']]

age['total_suicides'] = age.groupby(['year','age'])['suicides_no'].transform('sum')

age.drop('suicides_no', axis=1, inplace=True)

age = age.drop_duplicates()

age = age[age['year']>=2000.0]

age = age.sort_values("year")

fig=px.bar(age,x='age', y="total_suicides", animation_frame="year", 

           animation_group="age", color="age", hover_name="age")

fig.show()
sns.scatterplot(data=data, x='suicides_no', y='population')

plt.title('Number of Suicides vs Population')
hdi = data.loc[:,['country','year','suicides_no','HDI for year']]

hdi['total_suicides'] = hdi.groupby('year')['suicides_no'].transform('sum')

hdi.drop('suicides_no', axis=1, inplace=True)



hdi['ratio'] = hdi['HDI for year']/hdi['total_suicides']

top_countries = ['Russian Federation','United States','Japan','France',"Ukraine"] 

hdi = hdi.loc[hdi['country'].isin(top_countries)]

hdi = hdi.drop_duplicates()

hdi = hdi[hdi['year']>=2000]

for country in top_countries:

    df = hdi[hdi['country']==country]

    sns.lineplot(data=df, x='year', y='ratio')
#lets do for russia at first

russia = data[data['country']=='Russian Federation'].copy()

gdp = russia.iloc[:,[1,4,9]].copy()

gdp['total_suicides'] = gdp.groupby('year')['suicides_no'].transform('sum')

gdp.drop('suicides_no', axis=1, inplace=True)

gdp = gdp.drop_duplicates()

fig = plt.figure(figsize=(20,7))

ax = fig.add_subplot(1,2,1)

ax.plot(gdp['year'], gdp.iloc[:,1])

ax.set_title('GDP per Year')

ax1 = fig.add_subplot(1,2,2)

ax1.plot(gdp['year'], gdp['total_suicides'])

ax1.set_title('Total numbe of Suicides per year')
gen = data.loc[:,['suicides_no','generation']]

gen['mean'] = gen.groupby('generation')['suicides_no'].transform('sum')

gen.drop('suicides_no', axis=1, inplace=True)

gen = gen.drop_duplicates()





fig = px.pie(gen, names='generation', values='mean', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
gen = data.loc[:,['suicides_no','generation']]

gen['mean'] = gen.groupby('generation')['suicides_no'].transform('mean')

gen.drop('suicides_no', axis=1, inplace=True)

gen = gen.drop_duplicates()





fig = px.pie(gen, names='generation', values='mean', template='seaborn')

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
fig = plt.figure(figsize=(20,7))

sns.scatterplot(data=data, x=' gdp_for_year ($) ', y='suicides_no')
fig = plt.figure(figsize=(20,7))

sns.scatterplot(data=data, x='gdp_per_capita ($)', y='suicides_no')
fig = plt.figure(figsize=(20,7))

sns.scatterplot(data=data, x='suicides_no', y='suicides/100k pop')

plt.title('Number of suicides vs for 100k population')