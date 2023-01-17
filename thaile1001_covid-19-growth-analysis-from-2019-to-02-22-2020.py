# Let's import some essential packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import folium # For creating map



print('Packages imported')
# Let's get some new data from the sharing from github

# Start with the comfirmed situations

confirmed = pd.read_csv("../input/corona0221202/time_series_19-covid-Confirmed.csv")

confirmed.head()
# Death  

death = pd.read_csv("../input/corona0221202/time_series_19-covid-Deaths.csv")

death.head()
# Recovered

recovered = pd.read_csv("../input/corona0221202/time_series_19-covid-Recovered.csv")

recovered.head()
print('Confirmed situations')

confirmed.isna().sum()[confirmed.isna().sum()>0]

print('Death situations')

death.isna().sum()[confirmed.isna().sum()>0]
print('Recovered situations')

recovered.isna().sum()[confirmed.isna().sum()>0]
# Fill `unknown` with missing values

confirmed = confirmed.fillna('unknown')

death = death.fillna('unknown')

recovered = recovered.fillna('unknown')

print('All N/A filled with `unknown`')
last_date = '2/21/20'
china = confirmed[['Province/State',last_date]][confirmed['Country/Region']=='Mainland China']

china['death'] = death[last_date][death['Country/Region']=='Mainland China']

china['recovered'] = recovered[last_date][recovered['Country/Region']=='Mainland China']



# Bring `Province/State` to index

china = china.set_index('Province/State')



# Rename the columns

china = china.rename(columns = {last_date:'confirmed','death':'death','recovered':'recovered'})
# Let's see what it becomes

china.head()
# Plot data

china.sort_values(by='confirmed'

                  ,ascending=True).plot(kind='barh'

                                    , figsize=(20,30)

                                    , color = ['blue','red','lime']

                                    , width=1

                                    , rot=2)



# Represent the legends and titles

plt.title('Total cases by Province/State of Chinese Mainland', size = 40)

plt.ylabel('Province/State', size = 30)

plt.legend(bbox_to_anchor=(0.95,0.95) # setting coordinates for the caption box

           , frameon = True

           , fontsize = 20

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
hubei = china[china.index=='Hubei']

hubei = hubei.iloc[0]



# Let's plot

plt.figure(figsize=(15,15))



# Get the percentage from all Hubei's cases

hubei.plot(kind='pie'

          , colors=['#4b8bbe', 'red', 'lime']

          , autopct='%1.1f%%' # Add the %

          , shadow=True

          , startangle=140)



plt.title('Case distribution of Hubei', size=30)

plt.legend(loc = 'upper right'

          , frameon = True

          , fontsize = 15

          , ncol = 2

          , fancybox = True

          , framealpha = 0.95

          , shadow = True

          , borderpad = 1)
# Create a subset with confirmed cases in China

confirmed_china = confirmed[confirmed['Country/Region']=='Mainland China']

confirmed_china = confirmed_china.groupby(confirmed_china['Country/Region']).sum()



# Get confirmed cases growth over the time

confirmed_china = confirmed_china.iloc[0][2:confirmed_china.shape[1]]



# From stat do plot

plt.figure(figsize=(20,10))

plt.plot(confirmed_china

        , color = 'blue'

        , label = 'comfirmed'

        , marker = 'o')



plt.title('Growth of confirmed cases in Mainland China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=45,size=15)

plt.yticks(size=15)



plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)
# Create a subset with recoreved cases in China

recovered_china = recovered[recovered['Country/Region']=='Mainland China']

recovered_china = recovered_china.groupby(recovered_china['Country/Region']).sum()



# Get recovered cases growth over the time

recovered_china = recovered_china.iloc[0][2:recovered_china.shape[1]]



# Create a subset with death cases in China

death_china = death[death['Country/Region']=='Mainland China']

death_china = death_china.groupby(death_china['Country/Region']).sum()



# Get death cases growth over the time

death_china = death_china.iloc[0][2:death_china.shape[1]]



# Plot again

plt.figure(figsize=(20,10))



# Create a lineplot for each case variable(suspected, recovered and death)

plt.plot(recovered_china

        , color = 'lime'

        , label = 'recovered'

        , marker = 'o')



plt.plot(death_china

        , color = 'red'

        , label = 'death'

        , marker = 'o')



plt.title('Recovered vs Deaths Growth in Mainland China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=45,size=15)

plt.yticks(size=15)



plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)
# Select cases not in China

other_countries = confirmed[['Country/Region','Province/State',last_date]][confirmed['Country/Region']!='Mainland China']

other_countries['death'] = death[last_date][death['Country/Region']!='Mainland China']

other_countries['recovered'] = recovered[last_date][recovered['Country/Region']!='Mainland China']



# Sum the cases by Country/Region

other_countries = other_countries.groupby(other_countries['Country/Region']).sum()



# Rename the columns

other_countries = other_countries.rename(columns = {last_date:'confirmed','death':'death','recovered':'recovered'})



# creating the plot

other_countries.sort_values(by='confirmed'

                            ,ascending=True).plot(kind='barh'

                                            , figsize=(20,30)

                                            , color = ['blue','red','lime']

                                            , width=1

                                            , rot=2)



# defyning titles, labels, xticks and legend parameters

plt.title('Total cases in other countries', size=40)

plt.ylabel('country',size=30)

plt.yticks(size=20)

plt.xticks(size=20)

plt.legend(bbox_to_anchor=(0.95,0.95)

           , frameon = True

           , fontsize = 20

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)
# Get cases in South Korea

South_Korea = other_countries[other_countries.index=="South Korea"]

South_Korea = South_Korea.iloc[0]



# Get cases in Japan

Japan = other_countries[other_countries.index=="Japan"]

Japan = Japan.iloc[0]



# Get cases in Singapore

Singapore = other_countries[other_countries.index=="Singapore"]

Singapore = Singapore.iloc[0]



# Get cases in Hong-Kong

Hong_Kong = other_countries[other_countries.index=="Hong Kong"]

Hong_Kong = Hong_Kong.iloc[0]



fig, axes = plt.subplots(

                     ncols=2,

                     nrows=2,

                     figsize=(15, 15))



ax1, ax2, ax3, ax4 = axes.flatten()



# Add the percentage

ax1.pie(South_Korea

        , colors=['#4b8bbe','red','lime']

        , autopct='%1.1f%%' # Add percentagens

        , labels=['confirmed','death','recovered']

        , shadow=True

        , startangle=140)

ax1.set_title("South Korea Cases Distribution")



ax2.pie(Japan

           , colors=['#4b8bbe','red','lime']

           , autopct='%1.1f%%'

           , labels=['confirmed','death','recovered']

           , shadow=True

           , startangle=140)

ax2.set_title("Japan Cases Distribution")



ax3.pie(Singapore

           , colors=['#4b8bbe','red','lime']

           , autopct='%1.1f%%' 

           , labels=['confirmed','death','recovered']

           , shadow=True

           , startangle=140)

ax3.set_title("Singapore Cases Distribution")



ax4.pie(Hong_Kong

           , colors=['#4b8bbe','red','lime']

           , autopct='%1.1f%%' 

           , labels=['confirmed','death','recovered']

           , shadow=True

           , startangle=140)

ax4.set_title("Hong-Kong Cases Distribution")



fig.legend(['confirmed','death','recovered']

           , loc = "upper right"

           , frameon = True

           , fontsize = 15

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)



plt.show()
# Get cses by country that are not located in Mainland China

other_countries_confirmed = confirmed[confirmed.columns[4:confirmed.shape[1]]][confirmed['Country/Region']!='Mainland China']

other_countries_confirmed = other_countries_confirmed.iloc[0:other_countries_confirmed.shape[0]].sum()



other_countries_death = death[death.columns[4:death.shape[1]]][death['Country/Region']!='Mainland China']

other_countries_death = other_countries_death.iloc[0:other_countries_death.shape[0]].sum()



other_countries_recovered = recovered[recovered.columns[4:recovered.shape[1]]][recovered['Country/Region']!='Mainland China']

other_countries_recovered = other_countries_recovered.iloc[0:other_countries_recovered.shape[0]].sum()



# Create a list with confirmed, recovered and deaths cases

list_of_tuples = list(zip(other_countries_confirmed, other_countries_death, other_countries_recovered)) 



# Create a dataframe with this list to plot the chart

other_countries_cases_growth = pd.DataFrame(list_of_tuples, index = other_countries_confirmed.index, columns = ['confirmed','death','recovered'])



# Create the plot

other_countries_cases_growth.plot(kind='bar'

                                  , figsize=(20,10)

                                  , width=1

                                  , color=['#4b8bbe','red','lime']

                                  , rot=2)



# defyning title, labels, ticks and legend parameters

plt.title('Growth of cases over the days in other countries', size=30)

plt.xlabel('Updates', size=20)

plt.ylabel('Cases', size=20)

plt.xticks(rotation=45, size=15)

plt.yticks(size=15)

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)
# taking mortality and recovered ratios in China

recovered_rate = (recovered_china/(confirmed_china+death_china+recovered_china))*100

mortality_rate = (death_china/(confirmed_china+death_china+recovered_china))*100



# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot for each case variable(suspected, recovered and death)

plt.plot(recovered_rate

        , color = 'lime'

        , label = 'recovered rate'

        , marker = 'o')



plt.plot(mortality_rate

        , color = 'red'

        , label = 'death rate'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Mortality vs Recovered Rate Over the time In Mainland China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=45,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)
recovered_rate_other_countries = (other_countries_recovered/(other_countries_recovered+other_countries_death+other_countries_confirmed))*100

death_rate_other_countries = (other_countries_death/(other_countries_recovered+other_countries_death+other_countries_confirmed))*100



# defyning plotsize

plt.figure(figsize=(20,10))



# creating a lineplot for each case variable(suspected, recovered and death)

plt.plot(recovered_rate_other_countries

        , color = 'lime'

        , label = 'recovered rate'

        , marker = 'o')



plt.plot(death_rate_other_countries

        , color = 'red'

        , label = 'death rate'

        , marker = 'o')



# defyning titles, labels and ticks parameters

plt.title('Mortality vs Recovered Rates Over the time outside of China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=45,size=15)

plt.yticks(size=15)



# defyning legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1)
data = pd.read_csv("../input/corona0221202/02-21-2020.csv")

data.head()
# Creating a dataframe with total no of confirmed cases for every country

Number_of_countries = len(data['Country/Region'].value_counts())





cases = pd.DataFrame(data.groupby('Country/Region')['Confirmed'].sum())

cases['Country/Region'] = cases.index

cases.index=np.arange(1,Number_of_countries+1)



global_cases = cases[['Country/Region','Confirmed']]

#global_cases.sort_values(by=['Confirmed'],ascending=False)

global_cases
# Import the world_coordinates dataset

world_coordinates = pd.read_csv('../input/world-coordinates/world_coordinates.csv')

world_coordinates.head()
# Rename the columns

world_coordinates = world_coordinates.rename(columns = {'Code':'Code', 'Country':'Country/Region','latitude':'latitude','longitude':'longitude'})



# Merging the coordinates dataframe with original dataframe

world_data = pd.merge(world_coordinates,global_cases,on='Country/Region')

world_data.head()
# Create map and display it

world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')



for lat, lon, value, name in zip(world_data['latitude'], world_data['longitude'], world_data['Confirmed'], world_data['Country/Region']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country/Region</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'), # This is for displaying info when clicking on the red circle

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map)

world_map
# A look at the different cases - confirmed, death and recovered

print('Globally Confirmed Cases: ',data['Confirmed'].sum())

print('Global Deaths: ',data['Deaths'].sum())

print('Globally Recovered Cases: ',data['Recovered'].sum())
# Let's look the various Provinces/States affected

data.groupby(['Country/Region','Province/State']).sum()
# Provinces where deaths have taken place

data.groupby('Country/Region')['Deaths'].sum().sort_values(ascending=False)[:5]
# Lets also look at the Recovered stats

data.groupby('Country/Region')['Recovered'].sum().sort_values(ascending=False)[:5]
China = data[data['Country/Region']=='Mainland China']

China
f, ax = plt.subplots(figsize=(12, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Province/State", data=China[1:],

            label="Confirmed", color="r")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Province/State", data=China[1:],

            label="Recovered", color="g")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 400), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
latitude = 39.91666667

longitude = 116.383333

 

# create map and display it

china_map = folium.Map(location=[latitude, longitude], zoom_start=12)



china_coordinates= pd.read_csv("../input/chinese-cities/china_coordinates.csv")

china_coordinates.head()
china_coordinates.rename(columns={'city':'Province/State','lat':'latitude','lng':'longitude'},inplace=True)

df_china_virus = China.merge(china_coordinates)



# Make a data frame with dots to show on the map

data = pd.DataFrame({

   'name':list(df_china_virus['Province/State']),

   'lat':list(df_china_virus['latitude']),

   'lon':list(df_china_virus['longitude']),

   'Confirmed':list(df_china_virus['Confirmed']),

   'Recovered':list(df_china_virus['Recovered']),

   'Deaths':list(df_china_virus['Deaths'])

})



data.head()
# create map for total confirmed cases in china till date

china_map1 = folium.Map(location=[latitude, longitude], zoom_start=4,tiles='Stamen Toner')



for lat, lon, value, name in zip(data['lat'], data['lon'], data['Confirmed'], data['name']):

    folium.CircleMarker([lat, lon],

                        radius=13,

                        popup = ('Province: ' + str(name).capitalize() + '<br>'

                        'Confirmed: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(china_map1)

    folium.Map(titles='jj', attr="attribution")    

china_map1
china_map = folium.Map(location=[latitude, longitude], zoom_start=4,tiles='Stamen Toner')



for lat, lon, value, name in zip(data['lat'], data['lon'], data['Deaths'], data['name']):

    folium.CircleMarker([lat, lon],

                        radius=value*0.2,

                        popup = ('Province: ' + str(name).capitalize() + '<br>'

                        'Deaths: ' + str(value) + '<br>'),

                        color='black',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(china_map)

    folium.Map(titles='jj', attr="attribution")    

china_map
china_map = folium.Map(location=[latitude, longitude], zoom_start=4,tiles='Stamen Toner')



for lat, lon, value, name in zip(data['lat'], data['lon'], data['Recovered'], data['name']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('Province: ' + str(name).capitalize() + '<br>'

                        'Recovered: ' + str(value) + '<br>'),

                        color='green',

                        

                        fill_color='green',

                        fill_opacity=0.7 ).add_to(china_map)

       

china_map