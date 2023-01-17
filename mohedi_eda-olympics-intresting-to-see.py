import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly
normal_summer=pd.read_csv('../input/olympic-games/summer.csv')

winter=pd.read_csv('../input/olympic-games/winter.csv')

dic=pd.read_csv('../input/olympic-games/dictionary.csv')
normal_summer.info()
dic.head()
normal_summer.rename(columns={'Country':'Code'},inplace=True)
normal_summer=pd.merge(normal_summer,dic,on='Code',how='outer')
normal_summer.head()
normal_summer.describe()
normal_summer.describe(include=['O'])
normal_summer.head()
medals_map=normal_summer.groupby(['Country','Code'])['Medal'].count().reset_index()

medals_map=medals_map[medals_map['Medal']>0]



fig = px.choropleth(medals_map, locations="Code",

                    color="Medal", # lifeExp is a column of gapminder

                    hover_name="Country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()
normal_summer['useless']=1
medals_per_country=pd.pivot_table(index='Code',columns='Medal',values='useless',data=normal_summer,aggfunc=sum).fillna(0)
medals_per_country['Total']=medals_per_country['Gold']+medals_per_country['Silver']+medals_per_country['Bronze']
medals_per_country.sort_values(by='Total',ascending=False,inplace=True)
top=medals_per_country[:10]
top[['Bronze','Gold','Silver']].plot.barh(width=0.8,color=['#CD7F32','#FFDF00','#D3D3D3'])

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.title('Medals Distribution Of Top 10 Countries (Winter Olympics)')

plt.show()
medals_per_year=pd.pivot_table(index='Year',columns='Medal',values='useless',data=normal_summer,aggfunc=sum)
medals_per_year.plot(color=['#CD7F32','#FFDF00','#D3D3D3'],figsize=(15,6))

plt.title('Medals per Year',fontsize=15)

plt.ylabel('Number of Medals',fontsize=15)

plt.xlabel('Year',fontsize=15)
disciplines_per_year=pd.pivot_table(index='Year',columns='Discipline',values='useless',data=normal_summer).fillna(0)
disciplines_per_year['disciplines']=0

for col in disciplines_per_year.columns:

    disciplines_per_year['disciplines']+=disciplines_per_year[col]
disciplines_per_year.head()
disciplines_per_year['disciplines'].plot(figsize=(15,6))

plt.ylabel('Disciplines',fontsize=15)

plt.xlabel('Year',fontsize=15)

plt.title('Count of discplines per year',fontsize=15)
df=pd.merge(medals_per_country.reset_index(),normal_summer[['Code','Population','GDP per Capita']],on='Code').drop_duplicates().reset_index().drop('index',axis=1)
df.head()
df.corr()
medals_country_sport=pd.pivot_table(index='Code',columns='Sport',values='useless',data=normal_summer,aggfunc=sum).fillna(0)


for col in medals_country_sport.columns:

    plt.figure()

    data=medals_country_sport[col].sort_values(ascending=False)[:10]

    sns.barplot(x=data.index,y=data)

normal_summer.head()
pd.pivot_table(index='Gender',columns='Medal',values='useless',aggfunc=sum,data=normal_summer)
pd.pivot_table(index='Year',columns='Gender',values='useless',aggfunc=sum,data=normal_summer).fillna(0).plot(color=['#33D7FF','#FF33E6'],figsize=(10,6))
medals_per_sport_per_gender=pd.pivot_table(columns='Sport',index='Gender',values='useless',aggfunc=sum,data=normal_summer).fillna(0)

for col in medals_per_sport_per_gender.columns:

    plt.figure()

    sns.barplot(y=medals_per_sport_per_gender[col],x=medals_per_sport_per_gender.index,palette='coolwarm')
the_era=normal_summer[normal_summer['Year'].apply(lambda x: x in [1920])]
the_era.info()
d=dict()

for i in range(1900,2013,4):

    d[i]=normal_summer[normal_summer['Year']==i]['Sport'].nunique()

    
pd.Series(d).plot(figsize=(10,6))

plt.xlabel('Year',fontsize=15)

plt.ylabel('Count of sports',fontsize=15)

plt.title('Count of sports per year',fontsize=15)
normal_summer['City'].nunique()
cities=pd.pivot_table(index='City',columns='Year',values='useless',data=normal_summer).fillna(0)
cities['Total']=0

for col in cities.columns:

    cities['Total']+=cities[col]

    
cities[cities['Total']>1]
cities[cities['Total']==1]