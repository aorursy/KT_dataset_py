# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

import folium

import geopandas as gp

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Continent_deaths=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/worldometer/worldometer-confirmed-cases-and-deaths-by-country-territory-or-conveyance.csv')

Continent_deaths=Continent_deaths.drop(Continent_deaths.iloc[:,0:1],axis=1)

Continent_deaths=Continent_deaths.drop(Continent_deaths.iloc[:,8:],axis=1)



Continent_deaths=Continent_deaths[Continent_deaths['country']!='Total:']

Continent_deaths
Continent=Continent_deaths.iloc[0:8].dropna()

Continent
continent=Continent['country']



total_cases12=Continent['total_cases']

total_recovered12=Continent['total_recovered']

total_deaths12=Continent['total_deaths']

plt.figure(figsize=(20,20))

plt.bar(continent,total_cases12,linewidth=2,color='blue')

plt.bar(continent,total_recovered12,linewidth=2,color='violet')

plt.bar(continent,total_deaths12,linewidth=2,color='yellow')

ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.box(on=None)
Country=Continent_deaths.iloc[8:]

Country.fillna(0,inplace=True)

Country
Country['total_cases']=Country['total_cases']-Country['total_cases'].mean()/Country['total_cases'].std()

Country['total_deaths']=Country['total_deaths']-Country['total_deaths'].mean()/Country['total_deaths'].std()

Country['total_recovered']=Country['total_recovered']-Country['total_recovered'].mean()/Country['total_recovered'].std()

Country['active_cases']=Country['active_cases']-Country['active_cases'].mean()/Country['active_cases'].std()

Country['serious_critical_cases']=Country['serious_critical_cases']-Country['serious_critical_cases'].mean()/Country['serious_critical_cases'].std()

Country
Cases=Country.set_index(['country'])

print('Country with most Cases is ',Cases['total_cases'].idxmax(),'with count of',Cases['total_cases'].max())

print('Country with most deaths is ', Cases['total_deaths'].idxmax(),'with the count of',Cases['total_deaths'].max())

print('Country with most Recoveries is ',Cases['total_recovered'].idxmax(),'with count of',Cases['total_recovered'].max())

print('Country with most Serious Cases is ',Cases['serious_critical_cases'].idxmax(),'with count of',Cases['serious_critical_cases'].max())

print('Country with min Cases is ',Cases['total_cases'].idxmin(),'with count of',Cases['total_cases'].min())

print('Country with min deaths is ', Cases['total_deaths'].idxmin(),'with the count of',Cases['total_deaths'].min())

print('Country with min Recoveries is ',Cases['total_recovered'].idxmin(),'with count of',Cases['total_recovered'].min())

print('Country with min Serious Cases is ',Cases['serious_critical_cases'].idxmin(),'with count of',Cases['serious_critical_cases'].min())

Country1=Country.iloc[0:50]

Country2=Country.iloc[50:100]

Country3=Country.iloc[100:170]

Country4=Country.iloc[170:]

country1=Country1['country']



total_cases1=Country1['total_cases']

total_recovered1=Country1['total_recovered']

total_deaths1=Country1['total_deaths']

plt.figure(figsize=(20,20))

plt.bar(country1,total_cases1,linewidth=2,color='blue')

plt.bar(country1,total_recovered1,linewidth=2,color='violet')

plt.bar(country1,total_deaths1,linewidth=2,color='yellow')

ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.box(on=None)
country2=Country2['country']



total_cases2=Country2['total_cases']

total_recovered2=Country2['total_recovered']

total_deaths2=Country2['total_deaths']

plt.figure(figsize=(20,20))

plt.bar(country2,total_cases2,linewidth=2,color='blue')

plt.bar(country2,total_recovered2,linewidth=2,color='violet')

plt.bar(country2,total_deaths2,linewidth=2,color='yellow')

ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.box(on=None)
country3=Country3['country']



total_cases3=Country3['total_cases']

total_recovered3=Country3['total_recovered']

total_deaths3=Country3['total_deaths']

plt.figure(figsize=(20,20))

plt.bar(country3,total_cases3,linewidth=2,color='blue')

plt.bar(country3,total_recovered3,linewidth=2,color='violet')

plt.bar(country3,total_deaths3,linewidth=2,color='yellow')

ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.box(on=None)
country4=Country4['country']



total_cases4=Country4['total_cases']

total_recovered4=Country4['total_recovered']

total_deaths4=Country4['total_deaths']

plt.figure(figsize=(20,20))

plt.bar(country4,total_cases4,linewidth=2,color='blue')

plt.bar(country4,total_recovered4,linewidth=2,color='violet')

plt.bar(country4,total_deaths4,linewidth=2,color='yellow')

ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.box(on=None)
san=r'/kaggle/input/world-countries/world-countries.json'

world=folium.Map(location=[51.579519,24.245497],tiles='Mapbox Bright',zoom_start=12)

folium.Choropleth(geo_data=san,data=Country,columns=['country','total_deaths'],key_on='feature.properties.name',fill_color='GnBu',fill_opacity=0.7,line_opacity=0.2).add_to(world)

world
san=r'/kaggle/input/world-countries/world-countries.json'

world1=folium.Map(location=[51.579519,24.245497],tiles='Mapbox Bright',zoom_start=12)

folium.Choropleth(geo_data=san,data=Country,columns=['country','total_cases'],key_on='feature.properties.name',fill_color='GnBu',fill_opacity=0.7,line_opacity=0.2).add_to(world1)

world1
san=r'/kaggle/input/world-countries/world-countries.json'

world3=folium.Map(location=[51.579519,24.245497],tiles='Mapbox Bright',zoom_start=12)

folium.Choropleth(geo_data=san,data=Country,columns=['country','total_recovered'],key_on='feature.properties.name',fill_color='PuBuGn',fill_opacity=0.7,line_opacity=0.2).add_to(world3)

world3
county=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv')

county.fillna(0,inplace=True)

county=county.drop('last_update',axis=1)

county
county.country_region.unique()
us_county=county[county['country_region']=='US']

us_county.head()
us1=us_county.set_index('province_state')

us1.head()
x=us1['confirmed']

y=us1['recovered']

c=us1['deaths']

z=(x-c)/x

print('Best Rate',z.idxmax(),'is',z.max())

print('Worst Rate',z.idxmin(),'is',z.min())

print('Most Affected US State',us1['deaths'].idxmax(),'with count of',us1['deaths'].max())

print('Least Affected US State',us1['deaths'].idxmin(),'with count of',us1['deaths'].min())
county=us_county['province_state']



confirmed=us_county['confirmed']

recovered=us_county['recovered']

deaths=us_county['deaths']

plt.figure(figsize=(20,20))

plt.bar(county,confirmed,linewidth=2,color='blue')

plt.bar(county,recovered,linewidth=2,color='green')

plt.bar(county,deaths,linewidth=2,color='red')



ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.title('US States')

plt.box(on=None)
county=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv')

county.fillna(0,inplace=True)

county=county.drop('last_update',axis=1)

county
canada_county=county[county['country_region']=='Canada']

canada_county
canada=canada_county.set_index('province_state')

canada
print('Most Affected Canada State',canada['deaths'].idxmax(),'with count of',canada['deaths'].max())

print('Least Affected Canada',canada['deaths'].idxmin(),'with count of',canada['deaths'].min())
x=canada['confirmed']

y=canada['recovered']

c=canada['deaths']

z=(x-c)/x

print('Best Rate',z.idxmax(),'is',z.max())

print('Worst Rate',z.idxmin(),'is',z.min())

county1=canada_county['province_state']



confirmed1=canada_county['confirmed']

recovered1=canada_county['recovered']

deaths1=canada_county['deaths']

plt.figure(figsize=(20,20))

plt.bar(county1,confirmed1,linewidth=2,color='blue')

plt.bar(county1,recovered1,linewidth=2,color='green')

plt.bar(county1,deaths1,linewidth=2,color='red')



ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.title('Canada States')

plt.box(on=None)
county=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv')

county.fillna(0,inplace=True)

county=county.drop('last_update',axis=1)

county.head()
china_county=county[county['country_region']=='China']

china_county.head()
china=china_county.set_index('province_state')

china
print('Most Affected China State',china['deaths'].idxmax(),'with count of',china['deaths'].max())

print('Least Affected China State',china['deaths'].idxmin(),'with count of',china['deaths'].min())
x=china['confirmed']

y=china['recovered']

c=china['deaths']

z=(x-c)/x

print('Best Rate',z.idxmax(),'is',z.max())

print('Worst Rate',z.idxmin(),'is',z.min())

county1=china_county['province_state']



confirmed1=china_county['confirmed']

recovered1=china_county['recovered']

deaths1=china_county['deaths']

plt.figure(figsize=(20,20))

plt.bar(county1,confirmed1,linewidth=2,color='blue')

plt.bar(county1,recovered1,linewidth=2,color='green')

plt.bar(county1,deaths1,linewidth=2,color='red')



ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.title('China States')

plt.box(on=None)
county=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv')

county.fillna(0,inplace=True)

county=county.drop('last_update',axis=1)

county
italy_county=county[county['country_region']=='Italy']

italy_county.head()
italy=italy_county.set_index('province_state')

italy
print('Most Affected Italy State',italy['deaths'].idxmax(),'with count of',italy['deaths'].max())

print('Least Affected Italy State',italy['deaths'].idxmin(),'with count of',italy['deaths'].min())
x=italy['confirmed']

y=italy['recovered']

c=italy['deaths']

z=(x-c)/x

print('Best Rate',z.idxmax(),'is',z.max())

print('Worst Rate',z.idxmin(),'is',z.min())

county1=italy_county['province_state']



confirmed1=italy_county['confirmed']

recovered1=italy_county['recovered']

deaths1=italy_county['deaths']

plt.figure(figsize=(20,20))

plt.bar(county1,confirmed1,linewidth=2,color='blue')

plt.bar(county1,recovered1,linewidth=2,color='green')

plt.bar(county1,deaths1,linewidth=2,color='red')



ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.title('Italy States')

plt.box(on=None)
county=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv')

county.fillna(0,inplace=True)

county=county.drop('last_update',axis=1)

county.head()
uk_county=county[county['country_region']=='United Kingdom']

uk_county.head()
uk=uk_county.set_index('province_state')

uk
print('Most Affected United Kingdom State',uk['deaths'].idxmax(),'with count of',uk['deaths'].max())

print('Least Affected United Kingdom',uk['deaths'].idxmin(),'with count of',uk['deaths'].min())
x=uk['confirmed']

y=uk['recovered']

c=uk['deaths']

z=(x-c)/x

print('Best Rate',z.idxmax(),'is',z.max())

print('Worst Rate',z.idxmin(),'is',z.min())

county1=uk_county['province_state']



confirmed1=uk_county['confirmed']

recovered1=uk_county['recovered']

deaths1=uk_county['deaths']

plt.figure(figsize=(20,20))

plt.bar(county1,confirmed1,linewidth=2,color='blue')

plt.bar(county1,recovered1,linewidth=2,color='green')

plt.bar(county1,deaths1,linewidth=2,color='red')



ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.title('UK States')

plt.box(on=None)
uk_cities=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/regional_sources/uk_government/covid-19-uk-historical-data.csv')

england_cities=uk_cities[uk_cities['country']=='England']

print(uk_cities.country.unique())

scotland_cities=uk_cities[uk_cities['country']=='Scotland']

wales_cities=uk_cities[uk_cities['country']=='Wales']

ireland_cities=uk_cities[uk_cities['country']=='Northern Ireland']
print('Total Cases in England are',england_cities['totalcases'].sum())

print('Total Cases in Wales are',wales_cities['totalcases'].sum())

print('Total Cases in Scotland are',scotland_cities['totalcases'].sum())

print('Total Cases in Northern Ireland are',ireland_cities['totalcases'].sum())
UK_cases=pd.DataFrame({'Country':['England','Scotland','Wales','Northern Ireland'],

                   'Total Cases':[england_cities['totalcases'].sum(),scotland_cities['totalcases'].sum(),wales_cities['totalcases'].sum(),

                                 ireland_cities['totalcases'].sum()]})

UK_cases.head()
bars=plt.bar(UK_cases['Country'],UK_cases['Total Cases'],linewidth=2,color='orange')

ax=plt.gca()

plt.box(on=None)

for bar in bars:

    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+4,str(int(bar.get_height())),ha='center',fontsize=12,color='black')

ax.legend(['Cases'],loc='best',frameon=False)

ax.tick_params(left='on',bottom='on')
county=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv')

county.fillna(0,inplace=True)

county=county.drop('last_update',axis=1)

county.head()
germany_county=county[county['country_region']=='Germany']

germany_county.head()
germany=germany_county.set_index('province_state')

germany
print('Most Affected Germnay State',germany['deaths'].idxmax(),'with count of',germany['deaths'].max())

print('Least Affected Germany',germany['deaths'].idxmin(),'with count of',germany['deaths'].min())
x=germany['confirmed']

y=germany['recovered']

c=germany['deaths']

z=(x-c)/x

print('Best Rate',z.idxmax(),'is',z.max())

print('Worst Rate',z.idxmin(),'is',z.min())

county1=germany_county['province_state']



confirmed1=germany_county['confirmed']

recovered1=germany_county['recovered']

deaths1=germany_county['deaths']

plt.figure(figsize=(20,20))

plt.bar(county1,confirmed1,linewidth=2,color='blue')

plt.bar(county1,recovered1,linewidth=2,color='green')

plt.bar(county1,deaths1,linewidth=2,color='red')



ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.title('Germany States')

plt.box(on=None)
county=pd.read_csv('/kaggle/input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv')

county.fillna(0,inplace=True)

county=county.drop('last_update',axis=1)

county.head()
australia_county=county[county['country_region']=='Australia']

australia_county.head()
australia=australia_county.set_index('province_state')

australia
print('Most Affected Australia State',australia['deaths'].idxmax(),'with count of',australia['deaths'].max())

print('Least Affected Australia State',australia['deaths'].idxmin(),'with count of',australia['deaths'].min())
x=australia['confirmed']

y=australia['recovered']

c=australia['deaths']

z=(x-c)/x

print('Best Rate',z.idxmax(),'is',z.max())

print('Worst Rate',z.idxmin(),'is',z.min())

county1=australia_county['province_state']



confirmed1=australia_county['confirmed']

recovered1=australia_county['recovered']

deaths1=australia_county['deaths']

plt.figure(figsize=(20,20))

plt.bar(county1,confirmed1,linewidth=2,color='blue')

plt.bar(county1,recovered1,linewidth=2,color='green')

plt.bar(county1,deaths1,linewidth=2,color='red')



ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total Cases','Total Recovered','Total Deaths'],loc='best')

plt.title('Australia States')

plt.box(on=None)
import folium



folium.Marker([42.1657,-74.9481], popup='<i>US Most Affected State New York</i>', icon=folium.Icon(icon='info-sign')).add_to(world)

folium.Marker([-14.2710,-170.1320], popup='<i>US Least Affected State American Samoa</i>', icon=folium.Icon(color='green')).add_to(world)

folium.Marker([52.9399,-73.5491], popup='<i>Canada Most Affected State Quebec</i>', icon=folium.Icon(color='red')).add_to(world)

folium.Marker([46.5653,-66.4619], popup='<i>Canada Least Affected State New Brunswick</i>', icon=folium.Icon(icon='cloud',color='orange')).add_to(world)

folium.Marker([30.9756,112.2707], popup='<i>China Most Affected State  Hubei</i>', icon=folium.Icon(color='purple')).add_to(world)

folium.Marker([32.9711,119.4550], popup='<i>China least Affected State Jiangsu </i>', icon=folium.Icon(icon='cloud',color='blue')).add_to(world)

folium.Marker([45.466794,9.190347], popup='<i>Italy Most Affected State Lombardia </i>', icon=folium.Icon(color='pink')).add_to(world)

folium.Marker([41.557748,14.659161], popup='<i>Italy least Affected State Molise </i>', icon=folium.Icon(icon='cloud',color='lightred')).add_to(world)

folium.Marker([49.372300,-2.364400], popup='<i>UK most Affected State Channel Islands </i>', icon=folium.Icon(color='lightblue')).add_to(world)

folium.Marker([18.220600,-63.068600], popup='<i>Uk least Affected State Anguilla </i>', icon=folium.Icon(icon='cloud',color='cadetblue')).add_to(world)

folium.Marker([48.7904,11.4979], popup='<i>Germany Most Affected State Bayern </i>', icon=folium.Icon(color='black')).add_to(world)

folium.Marker([0.0000,0.0000], popup='<i>Germany least Affected State</i>', icon=folium.Icon(icon='cloud',color='cadetblue')).add_to(world)

folium.Marker([-33.8688,151.2093], popup='<i>Australia Most Affected State New South Wales</i>', icon=folium.Icon(color='gray')).add_to(world)

folium.Marker([-12.4634,130.8456], popup='<i>Australia least Affected State Northern Territory </i>', icon=folium.Icon(icon='cloud',color='beige')).add_to(world)



world
us_county=us_county[us_county['people_tested']>0]

us_county
us_county=us_county[us_county['deaths']>5]

us_county
us_county=us_county.reset_index()

us_county=us_county.drop(['index'],axis=1)

us_county
beds=pd.read_csv('/kaggle/input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-20-population-contracted.csv')

beds.fillna(0,inplace=True)

beds
us_beds=pd.merge(us_county,beds,how='right',left_index=True,right_index=True)



us_beds
us_beds=us_beds.drop(['uid','iso3'],axis=1)

us_beds
us_beds=us_beds.dropna()

us_beds
us_beds.info()
plt.figure(figsize=(15,15))

hos_beds=us_beds['total_icu_beds']

ava_beds=us_beds['available_icu_beds']

county=us_beds['province_state']

plt.bar(county,hos_beds,linewidth=2)

plt.bar(county,ava_beds,linewidth=2)

ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Total_beds','Available_beds'],loc='best')

plt.title('Total VS Available Beds')

plt.box(on=None)
US_beds=us_beds.set_index('province_state')

US_beds.head()
print('State with max cpacity of bed in hospitals is ',US_beds['total_icu_beds'].idxmax(),'with',int(US_beds['total_icu_beds'].max()),'beds')
plt.figure(figsize=(15,15))

infected=us_beds['projected_infected_individuals']

hospitalized=us_beds['proejcted_hospitalized_individuals']

county=us_beds['province_state']

plt.bar(county,infected,linewidth=2,color='red')

plt.bar(county,hospitalized,linewidth=2,color='yellow')

ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Hospiatlized Indiviuals','Infected Indivuals'],loc='best')

plt.title('Hospitalized VS Infected Indiviuals')

plt.box(on=None)
print('State with max Infected Indiviuals ',US_beds['projected_infected_individuals'].idxmax(),'with count of',int(US_beds['projected_infected_individuals'].max()))
plt.figure(figsize=(15,15))

hos_need=us_beds['hospital_beds_needed_six_months']

icu_need=us_beds['icu_beds_needed_six_months']

county=us_beds['province_state']

plt.bar(county,hos_need,linewidth=2,color='pink')

plt.bar(county,icu_need,linewidth=2,color='violet')

ax=plt.gca()

for x in ax.xaxis.get_ticklabels():

    x.set_rotation(90)

plt.legend(['Hospital beds needed for six months','Icu beds needed for six months'],loc='best')

plt.title('Beds needed for next six months')

plt.box(on=None)
print('State with max need of hosiptal beds for next six months is ',US_beds['hospital_beds_needed_six_months'].idxmax(),'with',int(US_beds['hospital_beds_needed_six_months'].max()),'beds')