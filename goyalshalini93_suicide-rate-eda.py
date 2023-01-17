#Filetring warnings

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
suicide_df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

suicide_df.head()
suicide_df.shape
suicide_df.describe(percentiles=[0.1,0.25,0.4,0.5,0.6,0.75,0.9,0.99,1])
count1 = len(suicide_df.loc[suicide_df['suicides_no'].between(np.percentile(suicide_df.suicides_no,0),

                                                     np.percentile(suicide_df.suicides_no,99))])

count2 = len(suicide_df.loc[suicide_df['suicides_no'].between(np.percentile(suicide_df.suicides_no,99),

                                                     np.percentile(suicide_df.suicides_no,100))])

print("Suicide Numbers between 0 and 99 percentile :",count1)

print("Suicide Numbers between 99 and 100 percentile :",count2)
suicide_df.info()
#Categorical variables

suicide_df.select_dtypes(include=[object]).head()
#Numerical variables

suicide_df.select_dtypes(exclude=[object]).head()
suicide_df.isnull().sum()
#Deleting 'HDI for year' column as most of the values are NaN

#Dropping column country-year as it is redundant

suicide_df.drop(['HDI for year','country-year'], axis=1, inplace=True)
#Checking columns with only one value throughout all the rows

suicide_df.loc[:,suicide_df.nunique()==1].columns
suicide_df.columns
#renaming columns for better readability and usability

suicide_df.rename(columns={'suicides/100k pop':'Suicides100kPop', ' gdp_for_year ($) ':'GDPForYear',

                          'gdp_per_capita ($)':'GDPPerCapita'}, inplace=True)
suicide_df.head()
suicide_df.GDPForYear = suicide_df.GDPForYear.apply(lambda x : x.replace(",", ""))

suicide_df.GDPForYear = suicide_df.GDPForYear.astype('int64')
#Changing GDPForYear into million $

suicide_df.GDPForYear = ((suicide_df.GDPForYear) / (1000000))

suicide_df.head()
plt.figure(figsize=(14,4))



plt.subplot(121)

plt.title('Suicide Number')

sns.distplot(suicide_df['suicides_no'], hist=False)



plt.subplot(122)

plt.title('Suicide Number per 100k population')

sns.distplot(suicide_df['Suicides100kPop'], hist=False)

plt.tight_layout()
lat_long = pd.read_csv('../input/country-geo/country_data.csv')

lat_long.rename(columns={'country':'countrycode','name':'country'},inplace=True)

lat_long.head()
#Checking if all the values in one Dataframe is present in other or not

temp1 = pd.DataFrame(suicide_df.country.unique())

temp2 = pd.DataFrame(lat_long.country.unique())

temp2.equals(temp1)
#Checking the values which are not present in lat_long

df = suicide_df.copy()

df = df.merge(lat_long, how = 'left', on = 'country')

df.loc[df.countrycode.isnull()].country.unique()
#Correcting the country names in our data set, then merging the dataset with lat_long

suicide_df.loc[df['country']=='Cabo Verde', 'country'] = 'Cape Verde'

suicide_df.loc[df['country']=='Republic of Korea', 'country'] = 'South Korea'

suicide_df.loc[df['country']=='Russian Federation', 'country'] = 'Russia'

suicide_df.loc[df['country']=='Saint Vincent and Grenadines', 'country'] = 'Saint Vincent and the Grenadines'

suicide_df = suicide_df.merge(lat_long[['latitude','longitude','country']], how = 'left', on = 'country')
#Adding the column suicide_country with total suicides in a country value

temp = suicide_df.copy()

table = temp.groupby(['country'])['suicides_no'].sum()

temp = temp.merge(table.reset_index(), how='left',on='country')

suicide_df['suicide_country'] = temp['suicides_no_y']

suicide_df.head()
#As we had seen earlier, the last percentile had a significant difference. 

#Looking at the countries included with the last percentile of suicides_no value.

suicide_df['country'].loc[suicide_df['suicides_no'].between(np.percentile(suicide_df.suicides_no,99),

                                                     np.percentile(suicide_df.suicides_no,100))].unique()
#Visualizing the top 10 countries with highest total suicide numbers

df = suicide_df.groupby(['country'])['suicides_no'].sum().sort_values(ascending=False).head(10)

df.plot.bar(figsize=(15,8))
#Visualizing the Top 5 countries with total suicides between 1985 to 2016, gender-wise against the total suicide number

plt.figure(figsize=(8,6))

df = suicide_df.loc[((suicide_df.country=='Russia') | (suicide_df.country=='United States')

                     | (suicide_df.country=='Japan') | (suicide_df.country=='France')

                    | (suicide_df.country=='Ukraine'))].groupby(['country','sex'])['suicides_no'].sum().unstack(fill_value=0).head(10)

df.plot.bar(figsize=(15,8))

#Visualizing the countries with total suicide counts on a map

from mpl_toolkits.basemap import Basemap



lat_min = min(suicide_df['latitude'])

lat_max = max(suicide_df['latitude'])

lon_min = min(suicide_df['longitude'])

lon_max = max(suicide_df['longitude'])



m = Basemap(

    projection='merc', 

    llcrnrlat=lat_min, 

    urcrnrlat=lat_max, 

    llcrnrlon=lon_min, 

    urcrnrlon=lon_max,

    resolution='l'

)

# Draw the components of the map



longitudes = suicide_df['longitude'].tolist()

latitudes = suicide_df['latitude'].tolist()

suicide_count = suicide_df['suicide_country'].values

fig = plt.figure(figsize=(30,30))

ax = fig.add_subplot(1,1,1)

ax = m.drawcountries()

ax = m.drawcoastlines(linewidth=0.1, color="white")

ax = m.fillcontinents(color='grey', alpha=0.6, lake_color='grey')

ax = m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

ax = m.scatter(longitudes, latitudes, c=suicide_count,s=500, zorder = 1,linewidth=1,latlon=True, edgecolors='yellow',cmap='YlOrRd'

               ,alpha=1)

plt.title('Suicide Number - Countrywise', fontsize=30)
plt.figure(figsize=(8,6))

df = suicide_df.loc[((suicide_df.country=='Russia') | (suicide_df.country=='United States')

                     | (suicide_df.country=='Japan') | (suicide_df.country=='France')

                    | (suicide_df.country=='Ukraine'))].groupby(['country','age'])['suicides_no'].sum().unstack(fill_value=0).head(10)

df.plot.bar(figsize=(15,8))
#Year against the total suicides that year, avg GDP and average total population of that year.

plt.figure(figsize=(10, 6))





df_time = suicide_df.groupby(["year"]).suicides_no.sum()

sns.lineplot(data = df_time)

plt.xlabel("Year")

plt.ylabel("Total Suicide Count")

plt.show()



#Year against suicide rate of the year Bar plot

df = suicide_df.groupby(['year'])['suicides_no'].sum()

df.plot(kind='bar',legend=True,figsize=(8,6),colormap='Pastel2')
print("Percent rows of 2015 :",round((len(suicide_df.loc[suicide_df.year==2015])/len(suicide_df.index))*100,2),"%")

print("Countries recorded in 2015 : ", len(suicide_df['country'].loc[suicide_df.year==2015].unique()))

print("Percent rows of 2016 :",round((len(suicide_df.loc[suicide_df.year==2016])/len(suicide_df.index))*100,2),"%")

print("Countries recorded in 2015 : ",len(suicide_df['country'].loc[suicide_df.year==2016].unique()))
# Seeing the total number of suicides in a country in a particular year

temp = suicide_df.copy()

table = temp.groupby(['country','year'])['suicides_no'].sum()

temp = temp.merge(table.reset_index(), how='left',on=['country','year'])

temp = temp.sort_values(by='suicides_no_y',ascending = False)

temp[['country','year','suicides_no_y']].drop_duplicates(keep='last').head(50)
#Finding out the countries with increasing suicide rate by year trend

def trend(countries,df):

    trend_up = pd.DataFrame()

    lst = []

    num = []

    for i in countries:

        cnt = 0

        rows = df.loc[df['country']==i]

        years = rows['year'].sort_values(ascending=False).unique()

        for j in years[:15]:

            suicide_year = rows['suicides_no_y'].loc[rows['year']==j].unique()

            suicide_year_prev = rows['suicides_no_y'].loc[rows['year']==(j-1)].unique()

            if(suicide_year > suicide_year_prev):

                cnt+=1

        if(cnt>=11):

            lst.append(i)

            num.append(cnt)

    trend_up['Count'] = num

    trend_up['Country'] = lst

    return trend_up.sort_values(by='Count',ascending=False)

                    

countries = temp['country'].unique()

df = temp[['country','year','suicides_no_y']]

lst = trend(countries,df)

lst
#Visualising the top five countries with an increasing suicide rate trend for past 15 years.

plt.figure(figsize=(8,6))

df = suicide_df.loc[((suicide_df.country=='United States') | (suicide_df.country=='Brazil')

                     | (suicide_df.country=='South Korea') | (suicide_df.country=='Mexico')

                    | (suicide_df.country=='Netherlands'))].groupby(['country','year'])['suicides_no'].sum().unstack(fill_value=0).head(10)

df.plot.bar(figsize=(15,8),legend=False,colormap='Accent')

#Seeing the countries with maximum number of suicides in 2015 

suicide_df[(suicide_df.year==2015)].groupby(['year','country'])['suicides_no'].sum().sort_values(ascending = False).head(10)
# A simple view of total number of suicides per age category in all the years from 1985 to 2016

df = suicide_df.groupby(['age'])['suicides_no'].sum().sort_values(ascending=False)

df.plot(kind='bar',legend=True,figsize=(8,6),colormap='Pastel1')
# Seeing the sex wise categorization of age in suicide numbers

df = suicide_df.groupby(['age','sex'])['suicides_no'].sum().unstack(fill_value=0)

df.plot(kind='bar',legend=True,figsize=(8,6),colormap='Pastel2')
sns.set(style="whitegrid")

ax = sns.violinplot(x=suicide_df["population"])
#In 2015 - most populous countries in the dataset

suicide_df[(suicide_df.year==2015)].groupby(['country'])['population'].sum().sort_values(ascending = False).head(10)
plt.figure(figsize=(15,5))

ax = sns.violinplot(x="age", y="population", data=suicide_df)
plt.figure(figsize=(15,5))

ax = sns.barplot(x="age", y="population", hue="sex", data=suicide_df, palette="muted")
plt.figure(figsize=(15,5))

ax = sns.barplot(y="generation", x="population",data=suicide_df)

suicide_df[(suicide_df.year==2015)].groupby(['year','country'])['Suicides100kPop'].sum().sort_values(ascending = False).head(20)
suicide_df.groupby(['year','country'])['Suicides100kPop'].sum().sort_values(ascending = False).head(20)
plt.figure(figsize=(15,6))



plt.subplot(121)

df_time = suicide_df.groupby(["year"]).GDPForYear.mean()

sns.lineplot(data = df_time)

plt.xlabel("Year")

plt.ylabel("Average GDP For Year")



plt.subplot(122)

df_time = suicide_df.groupby(["year"]).GDPPerCapita.mean()

sns.lineplot(data = df_time)

plt.xlabel("Year")

plt.ylabel("Average GDPPerCapita")

plt.tight_layout()

plt.show()

#Visualising the top five countries with an GDPPerCapita trend.

df = suicide_df.loc[((suicide_df.country=='United States') | (suicide_df.country=='Brazil')

                     | (suicide_df.country=='South Korea') | (suicide_df.country=='Mexico')

                    | (suicide_df.country=='Netherlands'))].groupby(['country','year'])['GDPPerCapita'].sum().unstack(fill_value=0).head(10)

df.plot.bar(figsize=(15,8),legend=False,colormap='Accent')

suicide_df.drop(['latitude', 'longitude', 'suicide_country'], axis=1,inplace=True)
sns.pairplot(suicide_df, hue="age")

plt.show()