import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from mpl_toolkits.basemap import Basemap
df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

df.rename(columns={'suicides_no': 'suicides'}, inplace=True)
df.shape
pd.options.display.max_rows = 999

df.head(1)
df.describe()
fig = plt.figure(figsize=(20,3))

sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
fig=plt.gcf()

fig.set_size_inches(8,8)

fig=sns.heatmap(df.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
concap = pd.read_csv('../input/world-capitals-gps/concap.csv')

print(concap.head())
by_year = df['suicides'].groupby([df['year'], df['sex']]).agg({'suicides':sum}).assign(percent = lambda x: 100 * x/x.sum())

by_year = np.round(by_year, decimals=2)

by_year = by_year.reset_index().sort_values(by='suicides', ascending=False)

most_year = by_year

print(most_year.head(5))

fig = plt.figure(figsize=(16,4))

ax = sns.lineplot(x="year", y="suicides",

hue="sex", style="sex",

markers=True, dashes=False, data=most_year)
suic_sex = df['suicides'].groupby(df['sex']).agg({'suicides' : 'sum'}).assign(percent = lambda x: 100 * x/x.sum())

suic_sex = np.round(suic_sex, decimals=0)

suic_sex = suic_sex.reset_index().sort_values(by='suicides',ascending=False)

most_sex = suic_sex

print("Total and percent of suicides among genders in year 1985 - 2016")

print()

print(most_sex)

fig = plt.figure(figsize=(6,4))

plt.title('Suicides by sex.')

sns.set(font_scale=0.9)

sns.barplot(y='suicides',x='sex',data=most_sex,palette="OrRd");

plt.ylabel('Number of suicides')

plt.tight_layout()
suic_age = df['suicides'].groupby(df['age']).agg({'suicides' : 'sum'}).assign(percent = lambda x: 100 * x/x.sum())

suic_age = np.round(suic_age, decimals=0)

suic_age = suic_age.reset_index().sort_values(by='suicides',ascending=False)

most_age = suic_age

print("Total and percent of suicides among age groups in year 1985 - 2016")

print()

print(most_age)

fig = plt.figure(figsize=(8,4))

plt.title('Suicides by age.')

sns.set(font_scale=0.9)

sns.barplot(y='suicides',x='age',data=most_age,palette="OrRd");

plt.ylabel('Number of suicides')

plt.tight_layout()
by_age_s = df['suicides'].groupby([df['sex'], df['age']]).agg({'suicides':sum}).assign(percent = lambda x: 100 * x/x.sum())

by_age_s = np.round(by_age_s, decimals=0)

by_age_s = by_age_s.reset_index().sort_values(by='suicides', ascending=False)

most_age_s = by_age_s

print("Total and percent of suicides among genders and age groups in year 1985 - 2016")

print()

print(most_age_s)

fig = plt.figure(figsize=(10,4))

plt.title('Suicides by age and sex')

sns.set(font_scale=0.9)

sns.barplot(y='suicides',x='age',hue='sex',data=most_age_s,palette='Set2');

plt.xlabel('Age and sex')

plt.ylabel('Suicides')

plt.tight_layout()
by_country = df['suicides'].groupby([df['country']]).agg({'suicides':'sum'}).assign(percent = lambda x: 100 * x/x.sum())

by_country = np.round(by_country, decimals=0)

by_country = by_country.reset_index().sort_values(by='suicides', ascending=False)

most_country = by_country.head(15)

print("Total and percent of suicides among countries in year 1985 - 2016")

print()

print(most_country.head(15))

fig = plt.figure(figsize=(20,6))

plt.title('Suicides by country')

sns.set(font_scale=0.9)

sns.barplot(y='suicides',x='country',data=most_country,palette='Set2');

plt.xlabel('Countries')

plt.ylabel('Suicides')

plt.tight_layout()



data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         by_country,left_on='CountryName',right_on='country')

def mapWorld(col1,size2,title3,label4,metr=100,colmap='hot'):

    m = Basemap(projection='mill',llcrnrlat=-60,urcrnrlat=70,\

            llcrnrlon=-110,urcrnrlon=180,resolution='c')

    m.drawcoastlines()

    m.drawcountries()

    m.drawparallels(np.arange(-90,91.,30.))

    m.drawmeridians(np.arange(-90,90.,60.))

    



    lat = data_full['CapitalLatitude'].values

    lon = data_full['CapitalLongitude'].values

    a_1 = data_full[col1].values

    if size2:

        a_2 = data_full[size2].values

    else: a_2 = 1



    m.scatter(lon, lat, latlon=True,c=a_1,s=metr*a_2,linewidth=1,edgecolors='black',cmap=colmap, alpha=1)

    

    cbar = m.colorbar()

    cbar.set_label(label4,fontsize=30)

    plt.title(title3, fontsize=30)

    plt.show()

sns.set(font_scale=1.5)

plt.figure(figsize=(20,10))

mapWorld(col1='suicides', size2=False,title3='Suicides by countries',label4='',metr=300,colmap='viridis')
by_pop = df['suicides/100k pop'].groupby(df['country']).agg({'suicides/100k pop':'mean'}).assign(percent = lambda x: 100 * x/x.sum())

by_pop = np.round(by_pop, decimals=0)

by_pop = by_pop.reset_index().sort_values(by='suicides/100k pop', ascending=False)

most_pop = by_pop.head(15)

print("Total and percent of suicides/100k pop in year 1985 - 2016")

print()

print(most_pop.head(15))

fig = plt.figure(figsize=(20,6))

plt.title('Suicides by country suicides/100k pop')

sns.set(font_scale=0.9)

sns.barplot(y='suicides/100k pop',x='country',data=most_pop,palette='Set2');

plt.xlabel('Countries')

plt.ylabel('suicides/100k pop')

plt.tight_layout()



data_full = pd.merge(concap[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

         by_pop,left_on='CountryName',right_on='country')

sns.set(font_scale=1.5)

plt.figure(figsize=(20,10))

mapWorld(col1='suicides/100k pop', size2=False,title3='Suicides by 100k pop',label4='',metr=300,colmap='viridis')
by_Lt = df.loc[df['country'] == 'Lithuania']

by_Lt = by_Lt['suicides/100k pop'].groupby([df['year'], df['sex']]).agg({'suicides/100k pop':'mean'}).assign(percent = lambda x: 100 * x/x.sum())

by_Lt = np.round(by_Lt, decimals=1)

by_Lt = by_Lt.reset_index().sort_values(by='year', ascending=True)

print("Total and percent of suicides/100k pop in Lithuania in year 1995 - 2016")

print()

print(by_Lt.head(10))





sns.set(style="whitegrid")

g = sns.catplot(x="year", y="suicides/100k pop", hue="sex", data=by_Lt,

                kind="bar", palette="muted", height=5, aspect=3)

g.despine(left=True)

plt.ylabel('suicides/100k pop')

plt.xlabel('Year')

plt.title('Suicides/100k pop in Lithuania')