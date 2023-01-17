#import libraries

import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import folium

from folium import plugins

from folium.plugins import HeatMap



from scipy.interpolate import interp1d

from scipy.interpolate import make_interp_spline, BSpline
df_global = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv")

df_countries_others = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")

df_states = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv")

df_major_cities = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByMajorCity.csv")

df_cities = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv")

df_emis = pd.read_csv("../input/co2-and-ghg-emission-data/emission data.csv")
#converting dates to the same format

df_global['dt'] = pd.to_datetime(df_global.dt)



#land average temperatures - some missing data

df_average_land = df_global.iloc[:, :3] # => 1750-2015

df_average_land = df_average_land.reset_index(drop=True)  #reset index



#land average temperature, minimum temperature, maximum temperature, land and ocean temperature (1850-2015) - no missing data

df_global = df_global.dropna(axis = 0)  #easy by deleting the rows with missing data => drop the first 100 years

df_global = df_global.reset_index(drop=True)  #reset index





#MISSING VALUES

#see what are the missing values for land temperatures dataset

null_data = df_average_land[df_average_land.isnull().any(axis=1)]

#print(null_data)  # => only the first 3 years have missing data

                  # we should delete the first 3 years but we are not working with this dataset





#see what are the missing values for land, min, max, land and occean

null_data2= df_global[df_global.isnull().any(axis=1)]

#print(null_data2)  # => no missing values

#separate the date column into day, month, year columns

df_global['day'] = df_global['dt'].dt.day

df_global['month'] = df_global['dt'].dt.month

df_global['year'] = df_global['dt'].dt.year



#grouping by year

earth_data = df_global.groupby(by = 'year')[['LandAverageTemperature', 'LandAverageTemperatureUncertainty',

       'LandMaxTemperature', 'LandMaxTemperatureUncertainty',

       'LandMinTemperature', 'LandMinTemperatureUncertainty',

       'LandAndOceanAverageTemperature',

       'LandAndOceanAverageTemperatureUncertainty']].mean().reset_index()



#create new column called 'turnpoint', which says for each date if it is before or after 1975

earth_data['turnpoint'] = np.where(earth_data['year'] <= 1975, 'before', 'after')





#2 subplots for land and land+ocean

fig = make_subplots(rows = 1, cols = 2)

fig.update_layout(title={'text': "Average Temperatures Before and After 1975", 'x':0.5, 'xanchor': 'center'}, 

                  font=dict( family="Times New Roman", size=20 ,color="white"), 

                  template = "plotly_dark", title_font_size = 25, title_font_family = "Times New Roman", hovermode= 'closest')



#boxplot for land average temperature

fig.add_trace(go.Box(x = earth_data['LandAverageTemperature'], y = earth_data['turnpoint'],boxpoints = 'all',jitter = 0.3, 

                     pointpos = -1.6, marker_color = 'rgb(255,160,122)', boxmean = True, name = 'Land'),

                     row = 1, col = 1)



#boxplot for land+ocean average temperature

fig.add_trace(go.Box(x = earth_data['LandAndOceanAverageTemperature'], y = earth_data['turnpoint'], boxpoints = 'all',jitter = 0.3, 

                     pointpos = -1.6, marker_color = 'rgb(32,178,170)', boxmean = True, name = 'Land and Ocean'),

                     row = 1, col = 2)



fig.update_traces(orientation='h')#horizontal orientation







#boxmean -  if True, we can see the mean for each box, as a line inside the box

#pointspos - sets the position of the sample points in relation to the box

        #  - if 0, the sample points are places over the center of the box

        #  - if negattive, points are under the box

#boxpoints - shows or not the outliers

#jitter -  the addition of a small amount of horizontal (or vertical) variability to the data in order to ensure 

           #all data points are visible,avoid overlapping
#convert to the same date format

df_countries_others['dt'] = pd.to_datetime (df_countries_others.dt)



#list containing the non-countries - collonies, atolls, autonomous regions, continents etc.

non_country_lst=['Antarctica', 'Africa', 'Asia', 'Europe', 'North America', 'South America','Denmark', 

                 'France', 'Netherlands','United Kingdom','Åland', 'American Samoa', 'Anguilla', 'Baker Island', 

                 'Bonaire', 'Saint Eustatius And Saba', 'British Virgin Islands', 'Cayman Islands', 

                 'Christmas Island', 'Falkland Islands (Islas Malvinas)', 'Faroe Islands', 'French Guiana', 

                 'French Southern And Antarctic Lands', 'Gaza Strip', 'Greenland', 'Guadeloupe', 'Guam', 'Guernsey', 

                 'Heard Island And Mcdonald Islands', 'Isle Of Man', 'Jersey', 'Kingman Reef', 'Macau', 'Martinique', 

                 'Mayotte', 'Montserrat', 'New Caledonia', 'Northern Mariana Islands', 'Palmyra Atoll', 'Reunion', 

                 'Saint Martin', 'Saint Pierre And Miquelon', 'South Georgia And The South Sandwich Islands', 

                 'Turks and Caicas Islands', 'Virgin Islands', 'Western Sahara']



#list containing the continents

continents = ['Antarctica', 'Africa', 'Asia', 'Australia', 'Europe', 'North America', 'South America']





#new dataset containing the CONTINENTS - even if we will not be using it here

df_continents = df_countries_others[df_countries_others.Country.isin(continents)]



#reseting the indexes of the continents dataset

df_continents.reset_index(drop=True)







#new dataset containing COUNTRIES

df_countries = df_countries_others[~df_countries_others.Country.isin(non_country_lst)]



#deleting Europe from the name of the actual countries

df_countries.loc[df_countries['Country'] == 'Denmark (Europe)', 'Country'] = 'Denmark'

df_countries.loc[df_countries['Country'] == 'France (Europe)', 'Country'] = 'France'

df_countries.loc[df_countries['Country'] == 'Netherlands (Europe)', 'Country'] = 'Netherlands'

df_countries.loc[df_countries['Country'] == 'United Kingdom (Europe)', 'Country'] = 'United Kingdom'



#reseting the indexes of the countries dataset

df_countries = df_countries.reset_index(drop=True)



#split the dt column into day, month, year

df_countries['day'] = df_countries['dt'].dt.day

df_countries['month'] = df_countries['dt'].dt.month

df_countries['year'] = df_countries['dt'].dt.year



#group by country and show from what year the recordings start

countries_min_year = df_countries.groupby(['Country']).min()



#since what year do all countries have a recording? => 1894

max(countries_min_year['year'].values)



#countries dataset all starting in 1894 and ending in 2012, because 2013 is incomplete

df_countries_1894 = df_countries.copy()

df_countries_1894 = df_countries_1894[(df_countries['year']>= 1894) & (df_countries['year'] < 2013)]



#MISSING DATA - 5 countries still have missing data in 1894-2012 interval - but we will not be using them

null_data_countries = df_countries_1894[df_countries_1894.isnull().any(axis=1)]

#null_data_countries.groupby(['Country']).count()



#reset index

df_countries_1894.reset_index(drop=True)



#list with all countries for which we have temperature recordings - we will use this later

countries_temp = df_countries_1894['Country'].unique()

#print(countries_temp)



#we use df_countries_1894 which has complete information in 2012 about all contained countries



countries = np.unique((df_countries_1894['Country']))

mean_temp = []

for country in countries:

    mean_temp.append(df_countries_1894[(df_countries_1894['Country'] == country) & (df_countries_1894['year'] == 2012)]['AverageTemperature'].mean())

    

    

data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_temp,

        locationmode = 'country names',

        text = countries,

        colorscale = [[0.0, "rgb(49,54,149)"],

                [0.1111111111111111, "rgb(69,117,180)"],

                [0.2222222222222222, "rgb(116,173,209)"],

                [0.3333333333333333, "rgb(171,217,233)"],

                [0.4444444444444444, "rgb(224,243,248)"],

                [0.5555555555555556, "rgb(254,224,144)"],

                [0.6666666666666666, "rgb(253,174,97)"],

                [0.7777777777777778, "rgb(244,109,67)"],

                [0.8888888888888888, "rgb(215,48,39)"],

                [1.0, "rgb(165,0,38)"]],

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Average\nTemperature,\n°C'),

            font=dict(family='Times New Roman', size=18, color='black')

            )

       ]



layout = dict(

    title = 'Average Land Temperature by Country in 2012',

    font=dict(family='Times New Roman', size=20, color='black'),

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(0,0,0)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')

    

#world emissions

df_emis_world = df_emis.copy()

df_emis_world = df_emis_world[df_emis_world['Country'] == 'World']

#df_emis_world



#countries emissions

df_emis_countries = df_emis.copy()

df_emis_countries = df_emis_countries[(df_emis_countries['Country'] != 'World') & (df_emis_countries['Country'] != 'Americas (other)') &

                                      (df_emis_countries['Country'] != 'Asia and Pacific (other)') & (df_emis_countries['Country'] != 'EU-28') &

                                      (df_emis_countries['Country'] != 'Europe (other)')]



#keep only the countries for which we have temperature recordings

df_emis_countries = df_emis_countries[(df_emis_countries.Country.isin(countries_temp))]

df_emis_countries = df_emis_countries.reset_index(drop=True)



#keep only 2007 - 2017 (no missing data)

df_emis_countries = df_emis_countries.drop(df_emis_countries.iloc[:, 1:257], axis = 1)



#new column growing rate of emissions = 100(max-min)/min

df_emis_countries['Growing Rate'] = ((df_emis_countries['2017'] - df_emis_countries['2007']) * 100) / df_emis_countries['2007']

#df_emis_countries
#df_emis_world #world emissions 1751 - 2017 but has strange format



world_emis = pd.read_csv("../input/world-emissions/world_emis_modif.csv") #transposed in excel

x_emis = world_emis['year']

y_emis = world_emis['emissions']



#font dictionary

font = {'family': 'serif',

        'color':  'white',

        'weight': 'normal',

        'size': 25

        }



#ticks font size

plt.rcParams['xtick.labelsize'] = 20

plt.rcParams['ytick.labelsize'] = 20



plt.style.use('dark_background')

plt.figure(figsize=(20,10))



plt.plot(x_emis,y_emis, color = "#FF7F50",  linewidth = 5)



#title

plt.title('Evolution of World Emissions 1751-2017', fontdict=font)

ax=plt.gca()

ax.title.set_position([.5, 1.1])



#labels

plt.xlabel('Years', fontdict=font)

plt.ylabel("Average Emissions (tonnes) / Year", fontdict=font)



plt.show()



#extracting the 2 datasets



#sorting ascending after growing rate of emissions

df_emis_countries = df_emis_countries.sort_values('Growing Rate', ascending=True)



#top 10 countries for growing rate of emissions

df_low_emis = df_emis_countries.head(10)

df_low_emis = df_low_emis.sort_values('Country', ascending=True)  #sort alfabetically

df_low_emis = df_low_emis.reset_index(drop=True)

df_low_emis.to_csv('low_emis.csv')



#bottom 10 countries for growing rate of emissions

df_high_emis = df_emis_countries.tail(10)

df_high_emis = df_high_emis.sort_values('Country', ascending=True)  #sort alfabetically

df_high_emis = df_high_emis.reset_index(drop=True)

df_high_emis.to_csv('high_emis.csv')
#importing the 2 datasets adjusted in excel and slecting the years 2007 and 2017



#top 10 countries for growing rate of emissions for 2007 and 2017

df_high_emis_2years = pd.read_csv("../input/emissions-sets-modified/high_emis_modif.csv") #transposed in excel

df_high_emis_2years = df_high_emis_2years[(df_high_emis_2years['Year'] == 2007) |

                                      (df_high_emis_2years['Year'] == 2017)]





#bottom 10 countries for growing rate of emissions for 2007 and 2017

df_low_emis_2years = pd.read_csv("../input/emissions-sets-modified/low_emis_modif.csv") #transposed in excel

df_low_emis_2years = df_low_emis_2years[(df_low_emis_2years['Year'] == 2007) |

                                      (df_low_emis_2years['Year'] == 2017)]
#font dictionary

font_title = {'family': 'serif',

        'color':  'white',

        'weight': 'normal',

        'size': 28

        }



#countries font

plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.size'] = '18'



#bar charts

fig, ax = plt.subplots(2,1, figsize = (10,10))







x_countries = df_high_emis_2years['Country'][df_high_emis_2years['Year'] == 2007]

y_2007 = df_high_emis_2years['Emissions'][df_high_emis_2years['Year'] == 2007]

y_2017 = df_high_emis_2years['Emissions'][df_high_emis_2years['Year'] == 2017]



ax[0].barh(x_countries,y_2017)

ax[0].barh(x_countries, y_2007)

ax[0].set_xlabel('GHG (tonnes)')

ax[0].set_ylabel('Country')



x_countries2 = df_low_emis_2years['Country'][df_low_emis_2years['Year'] == 2007]

y2_2007 = df_low_emis_2years['Emissions'][df_low_emis_2years['Year'] == 2007]

y2_2017 = df_low_emis_2years['Emissions'][df_low_emis_2years['Year'] == 2017]



ax[1].barh(x_countries2, y2_2017, label= '2017')

ax[1].barh(x_countries2, y2_2007, label= '2007')

ax[1].set_xlabel('GHG (tonnes)')

ax[1].set_ylabel('Country')



plt.legend()



ax=plt.gca()

ax.title.set_position([.5, 1.1])

plt.title('Growth Rate of Green House Gases', fontdict=font_title, x = 0.3, y=-0.3)



plt.tight_layout(pad=1.08, h_pad=2.08, rect=(0,0,1.4,1.4))

plt.show()
#plt.rcParams.update(plt.rcParamsDefault)
#average temperatures on years on countries

countries_average_year = df_countries_1894.copy()

countries_average_year = countries_average_year.groupby(['Country', 'year'])[['AverageTemperature']].mean().reset_index()



#10 countries with high emissions growth rate

countries_10_high = countries_average_year.copy()

countries_10_high = countries_10_high[(countries_10_high.Country == 'Afghanistan') | (countries_10_high.Country == 'Angola') |

                            (countries_10_high.Country == 'Benin') | (countries_10_high.Country == 'Bhutan') |

                            (countries_10_high.Country == 'Cambodia') | (countries_10_high.Country == 'Equatorial Guinea') |

                            (countries_10_high.Country == 'Liechtenstein') | (countries_10_high.Country == 'Namibia') |

                            (countries_10_high.Country == 'Nepal') | (countries_10_high.Country == 'Oman')]



countries_10_high = countries_10_high[(countries_10_high.year == 1894) | (countries_10_high.year == 1900) |

                            (countries_10_high.year == 1918) | (countries_10_high.year == 1930) |

                            (countries_10_high.year == 1941) | (countries_10_high.year == 1957) |

                            (countries_10_high.year == 1965) | (countries_10_high.year == 1979) |

                            (countries_10_high.year == 1995) | (countries_10_high.year == 2012)]

countries_10_high = countries_10_high.reset_index(drop=True)







#10 countries with low emissions growth rate

countries_10_low = countries_average_year.copy()

countries_10_low = countries_10_low[(countries_10_low.Country == 'Belgium') | (countries_10_low.Country == 'Denmark') |

                            (countries_10_low.Country == 'France') | (countries_10_low.Country == 'Germany') |

                            (countries_10_low.Country == 'Hungary') | (countries_10_low.Country == 'North Korea') |

                            (countries_10_low.Country == 'Romania') | (countries_10_low.Country == 'Sweden') |

                            (countries_10_low.Country == 'Ukraine') | (countries_10_low.Country == 'United Kingdom')]



countries_10_low = countries_10_low[(countries_10_low.year == 1894) | (countries_10_low.year == 1900) |

                            (countries_10_low.year == 1918) | (countries_10_low.year == 1930) |

                            (countries_10_low.year == 1941) | (countries_10_low.year == 1957) |

                            (countries_10_low.year == 1965) | (countries_10_low.year == 1979) |

                            (countries_10_low.year == 1995) | (countries_10_low.year == 2012)]

countries_10_low = countries_10_low.reset_index(drop=True)

#font dictionary

font = {'family': 'serif',

        'color':  'white',

        'weight': 'normal',

        'size': 25

        }



plt.figure(figsize=(12,12))

plt.title('Evolution of Average Temperature in Time', fontdict=font)

ax=plt.gca()

ax.title.set_position([.5, 1.1])



pivot_table = countries_10_high.pivot('Country', 'year', 'AverageTemperature')

sns.heatmap(pivot_table, annot=True, fmt=".1f", linewidths=.5, square=True, cmap='coolwarm',

           cbar_kws={'label': 'Average Temperature / Year'})

plt.xlabel('Years')

plt.show()
#font dictionary

font = {'family': 'serif',

        'color':  'white',

        'weight': 'normal',

        'size': 25

        }



plt.figure(figsize=(12,12))

plt.title('Evolution of Average Temperature in Time', fontdict=font)

ax=plt.gca()

ax.title.set_position([.5, 1.1])



pivot_table = countries_10_low.pivot('Country', 'year', 'AverageTemperature')

sns.heatmap(pivot_table, annot=True, fmt=".1f", linewidths=.5, square=True, cmap='coolwarm',

           cbar_kws={'label': 'Average Temperature / Year'})

plt.xlabel('Years')

plt.show()
#convert dates to the same format

df_major_cities['dt'] = pd.to_datetime(df_major_cities.dt)



#create three new columns for day, month and year

df_major_cities['day'] = df_major_cities['dt'].dt.day

df_major_cities['month'] = df_major_cities['dt'].dt.month

df_major_cities['year'] = df_major_cities['dt'].dt.year
#coldest cities in 2012 

#as proved below, common years for all 5 cities are 1855-2012



#see what are the cities

cities_grouped_min = df_major_cities[df_major_cities.year == 2012][['City','Country',

                    'AverageTemperature']].groupby(['City','Country']).mean().sort_values('AverageTemperature',ascending=True)[:5] #see what are the cities



#Harbin - complete years: 1820-2012

df_Harbin = df_major_cities[df_major_cities.City == 'Harbin']

df_Harbin = df_Harbin.reset_index(drop=True)  #rest index

#print(df_Harbin[df_Harbin.isnull().any(axis=1)].groupby(df_Harbin['year']).count())  #see years with missing data 

df_Harbin = df_Harbin[(df_Harbin['year'] >= 1855) & (df_Harbin['year'] <= 2012)]  #keep only common years

df_Harbin = df_Harbin.groupby('year')[['AverageTemperature', 'year']].mean()  #calculate yearly average temperature





#Saint Petersburg - complete years: 1753-2012

df_Saint_Petersburg = df_major_cities[df_major_cities.City == 'Saint Petersburg']

df_Saint_Petersburg = df_Saint_Petersburg.reset_index(drop=True)  #rest index

#print(df_Saint_Petersburg[df_Saint_Petersburg.isnull().any(axis=1)].groupby(df_Saint_Petersburg['year']).count()) #see years with missing data

df_Saint_Petersburg = df_Saint_Petersburg[(df_Saint_Petersburg['year'] >= 1855) & (df_Saint_Petersburg['year'] <= 2012)]  #keep only common years

df_Saint_Petersburg = df_Saint_Petersburg.groupby('year')[['AverageTemperature', 'year']].mean()  #calculate yearly average temperature





#Santiago - complete years: 1855-2012

df_Santiago = df_major_cities[df_major_cities.City == 'Santiago']

df_Santiago = df_Santiago.reset_index(drop=True)  #rest index

#print(df_Santiago[df_Santiago.isnull().any(axis=1)].groupby(df_Santiago['year']).count()) #see years with missing data

df_Santiago = df_Santiago[(df_Santiago['year'] >= 1855) & (df_Santiago['year'] <= 2012)]  #keep only common years

df_Santiago = df_Santiago.groupby('year')[['AverageTemperature', 'year']].mean()  #calculate yearly average temperature





#Changchun - complete years: 1833-2012

df_Changchun = df_major_cities[df_major_cities.City == 'Changchun']

df_Changchun = df_Changchun.reset_index(drop=True)  #rest index

#print(df_Changchun[df_Changchun.isnull().any(axis=1)].groupby(df_Changchun['year']).count()) #see years with missing data

df_Changchun = df_Changchun[(df_Changchun['year'] >= 1855) & (df_Changchun['year'] <= 2012)]  #keep only common years

df_Changchun = df_Changchun.groupby('year')[['AverageTemperature', 'year']].mean()  #calculate yearly average temperature





#Moscow - complete years: 1753-2012

df_Moscow = df_major_cities[df_major_cities.City == 'Moscow']

df_Moscow = df_Moscow.reset_index(drop=True)  #rest index

#print(df_Moscow[df_Moscow.isnull().any(axis=1)].groupby(df_Moscow['year']).count()) #see years with missing data

df_Moscow = df_Moscow[(df_Moscow['year'] >= 1855) & (df_Moscow['year'] <= 2012)]  #keep only common years

df_Moscow = df_Moscow.groupby('year')[['AverageTemperature', 'year']].mean()  #calculate yearly average temperature



#hottest cities in 2012

#as proved below, common years for all 5 cities are 1870-2012



#see what are the cities

cities_grouped_max = df_major_cities[df_major_cities.year == 2012][['City','Country',

                    'AverageTemperature']].groupby(['City','Country']).mean().sort_values('AverageTemperature',ascending=False).head()



#Umm Durman - complete years: 1870-2012

df_Umm_Durman = df_major_cities[df_major_cities.City == 'Umm Durman']

df_Umm_Durman = df_Umm_Durman.reset_index(drop=True)  #rest index

#print(df_Umm_Durman[df_Umm_Durman.isnull().any(axis=1)].groupby(df_Umm_Durman['year']).count()) #see years with missing data

df_Umm_Durman = df_Umm_Durman[(df_Umm_Durman['year'] >= 1870) & (df_Umm_Durman['year'] <= 2012)]  #keep only common years

df_Umm_Durman = df_Umm_Durman.groupby('year')[['AverageTemperature', 'year']].mean()  #calculate yearly average temperature



#Madras - complete years: 1865-2012

df_Madras = df_major_cities[df_major_cities.City == 'Madras']

df_Madras = df_Madras.reset_index(drop=True)  #rest index

#print(df_Madras[df_Madras.isnull().any(axis=1)].groupby(df_Madras['year']).count()) #see years with missing data

df_Madras = df_Madras[(df_Madras['year'] >= 1870) & (df_Madras['year'] <= 2012)]  #keep only common years

df_Madras = df_Madras.groupby('year')[['AverageTemperature', 'year']].mean()  #calculate yearly average temperature





#Bangkok - complete years: 1863-2012

df_Bangkok = df_major_cities[df_major_cities.City == 'Bangkok']

df_Bangkok = df_Bangkok.reset_index(drop=True)  #rest index

#print(df_Bangkok[df_Bangkok.isnull().any(axis=1)].groupby(df_Bangkok['year']).count()) #see years with missing data

df_Bangkok = df_Bangkok[(df_Bangkok['year'] >= 1870) & (df_Bangkok['year'] <= 2012)]  #keep only common years

df_Bangkok = df_Bangkok.groupby('year')[['AverageTemperature', 'year']].mean()  #calculate yearly average temperature





#Jiddah - complete years: 1864-2012

df_Jiddah = df_major_cities[df_major_cities.City == 'Jiddah']

df_Jiddah = df_Jiddah.reset_index(drop=True)  #rest index

#print(df_Jiddah[df_Jiddah.isnull().any(axis=1)].groupby(df_Jiddah['year']).count()) #see years with missing data

df_Jiddah = df_Jiddah[(df_Jiddah['year'] >= 1870) & (df_Jiddah['year'] <= 2012)]  #keep only common years

df_Jiddah = df_Jiddah.groupby('year')[['AverageTemperature', 'City', 'year']].mean()  #calculate yearly average temperature

 

    

#Ho Chi Minh City - complete years: 1863-2012

df_Ho_Chi_Minh = df_major_cities[df_major_cities.City == 'Ho Chi Minh City']

df_Ho_Chi_Minh = df_Ho_Chi_Minh.reset_index(drop=True)  #rest index

#print(df_Ho_Chi_Minh[df_Ho_Chi_Minh.isnull().any(axis=1)].groupby(df_Ho_Chi_Minh['year']).count()) #see years with missing data

df_Ho_Chi_Minh = df_Ho_Chi_Minh[(df_Ho_Chi_Minh['year'] >= 1870) & (df_Ho_Chi_Minh['year'] <= 2012)]  #keep only common years

df_Ho_Chi_Minh = df_Ho_Chi_Minh.groupby('year')[['AverageTemperature', 'year']].mean()  #calculate yearly average temperature

 
#coldest cities in 2012 



#Harbin

x_Harbin = df_Harbin['year']

y_Harbin = df_Harbin['AverageTemperature']



#Saint Petersburg

x_Saint_Petersburg = df_Saint_Petersburg['year']

y_Saint_Petersburg = df_Saint_Petersburg['AverageTemperature']



#Santiago

x_Santiago = df_Santiago['year']

y_Santiago = df_Santiago['AverageTemperature']



#Changchun

x_Changchun = df_Changchun['year']

y_Changchun = df_Changchun['AverageTemperature']



#Moscow

x_Moscow = df_Moscow['year']

y_Moscow = df_Moscow['AverageTemperature']





#plot

fig, [[ax1, ax2],[ax3, ax4]] = plt.subplots(2,2,)

fig.set_size_inches(18.5, 10.5)

ax1.plot(x_Harbin, y_Harbin, color = '#AFEEEE', linewidth = 3)

ax2.plot(x_Saint_Petersburg, y_Saint_Petersburg, color = '#48D1CC', linewidth = 3)

ax3.plot(x_Moscow,y_Moscow, color ='#20B2AA', linewidth = 3 )

ax4.plot(x_Changchun,y_Changchun, color = '#7FFFD4', linewidth = 3 )



fig.subplots_adjust(hspace=.5)

fig.suptitle('Global Warming - Coldest Cities in the World in 2012', fontsize=25)

fig.text(0.5, 0.04, 'Years', ha='center', va='center', fontsize=20)

fig.text(0.06, 0.5, 'Average Temperature / Year', ha='center', va='center', rotation='vertical', fontsize=20)



ax1.set_title('Habrin - China(N)', fontsize = 20)

ax2.set_title('Saint Petersburg - Rusia', fontsize = 20)

ax3.set_title('Moscow -Rusia', fontsize = 20)

ax4.set_title('Changchun - China', fontsize = 20)
#hotetst cities in 2012 



#Umm Durman

x_Umm_Durman = df_Umm_Durman['year']

y_Umm_Durman = df_Umm_Durman['AverageTemperature']



#Madras

x_Madras = df_Madras['year']

y_Madras = df_Madras['AverageTemperature']



#Bangkok

x_Bangkok = df_Bangkok['year']

y_Bangkok = df_Bangkok['AverageTemperature']



#Jiddah

x_Jiddah = df_Jiddah['year']

y_Jiddah = df_Jiddah['AverageTemperature']



#Ho Chi Minh City

x_Ho_Chi_Minh = df_Ho_Chi_Minh['year']

y_Ho_Chi_Minh = df_Ho_Chi_Minh['AverageTemperature']





#plot

fig, [[ax1, ax2],[ax3, ax4]] = plt.subplots(2,2,)

fig.set_size_inches(18.5, 10.5)

ax1.plot(x_Madras, y_Madras, color ='#FFB6C1', linewidth = 3)

ax2.plot(x_Ho_Chi_Minh, y_Ho_Chi_Minh, color = '#F08080', linewidth = 3)

ax3.plot(x_Umm_Durman,y_Umm_Durman, color ='#FFA07A', linewidth = 3 )

ax4.plot(x_Bangkok,y_Bangkok, '#DB7093', linewidth = 3 )



fig.subplots_adjust(hspace=.5)

fig.suptitle('Global Warming observed the hottest cities in the world', fontsize=20)

fig.text(0.5, 0.04, 'Years', ha='center', va='center', fontsize=20)

fig.text(0.06, 0.5, 'Average Temperature / Year', ha='center', va='center', rotation='vertical', fontsize=20)



ax1.set_title('Madras - India', fontsize=20)

ax2.set_title('Ho Chi Minh City - Vietnam', fontsize=20)

ax3.set_title('Durman - Sudan', fontsize=20)

ax4.set_title('Bangkok - Thailand', fontsize=20)
m = folium.Map(location = [34.047863, 100.619652], zoom_start = 3, tiles = 'Stamen Toner')



#hottest cities

folium.Circle(location = [13.082680, 80.270721], popup = 'Madras', radius = 140000, fill=True, fill_color='crimson', color='crimson'

 ).add_to(m)

folium.Circle(location = [10.823099, 106.629662], popup = 'Ho Chi Minh City',  radius = 140000 , fill=True, fill_color='crimson', color='crimson'

  ).add_to(m)

folium.Circle(location = [15.653120,32.481530], popup = 'Umm Durman',radius = 140000, fill=True, fill_color='crimson', color='crimson'

  ).add_to(m)

folium.Circle(location = [13.756331, 100.501762], popup = 'Bangkok', radius = 140000, fill=True, fill_color='crimson', color='crimson'

 ).add_to(m)





#coldest cities

folium.Circle(location =[45.803776, 126.534966], popup = 'Harbin', radius = 100000, fill=True, fill_color='blue', color='blue'

 ).add_to(m) 

folium.Circle(location =[59.938480, 30.312481], popup = 'Saint Petersburg', radius = 100000, fill=True, fill_color='blue', color='blue'

 ).add_to(m) 

folium.Circle(location =[55.755825, 37.617298], popup = 'Moscow', radius = 100000, fill=True, fill_color='blue', color='blue'

 ).add_to(m) 

folium.Circle(location =[43.817070, 125.323547], popup = 'Changchun', radius = 100000, fill=True, fill_color='blue', color='blue'

 ).add_to(m)



heat_data1 = [[45.803776, 126.534966, cities_grouped_min['AverageTemperature'][0] ]]

heat_data2 = [[59.938480, 30.312481, cities_grouped_min['AverageTemperature'][1] ]]

heat_data3 = [[55.755825, 37.617298, cities_grouped_min['AverageTemperature'][2] ]]

heat_data4 = [[55.755825, 37.617298, cities_grouped_min['AverageTemperature'][3] ]]

heat_data5 = [[15.653120,32.481530, cities_grouped_max['AverageTemperature'][0] ]]

heat_data6 = [[13.082680, 80.270721, cities_grouped_max['AverageTemperature'][1]]]

heat_data7 = [[13.756331, 100.501762, cities_grouped_max['AverageTemperature'][3]]]

heat_data8 = [[10.823099, 106.629662, cities_grouped_max['AverageTemperature'][4]]]

heat_data_cold = [heat_data1, heat_data2,heat_data3, heat_data4]

heat_data_hot =  [heat_data5, heat_data6, heat_data7, heat_data8]



for i in heat_data_cold:

    HeatMap(i).add_to(m)



for i in heat_data_hot:

    HeatMap(i,gradient = {.33: 'red', .66: 'brown', 1: 'green'}).add_to(m)

m
#import files

df = pd.read_csv("../input/world-emissions/Filtered_Major_Cities.csv")

co = pd.read_csv("../input/world-emissions/emis_cities.csv")

plt.style.use('dark_background')
#list_type variables holding the names of the cities and their colours that are to be attributed to the graph lines

cities = ['Guangzhou', 'Shanghai', 'Tangshan', 'Tianjin']

colour = ['#BC8F8F', '#F4A460', '#8FBC8F', '#FF6347']

i=0



#gets the data of a city from the temperature file

def get_city_from_tempcsv(city_name):

    """

    Returns a city object.

    """ 

    city = df.loc[(df['City'] == city_name)]

    city.reset_index(drop=True, inplace=True)

    return city



#gets the data of a city from the gasses file

def get_city_from_gascsv(city_name):

    """

    Returns a city object.

    """ 

    city = co.loc[(co['city'] == city_name)]

    city.reset_index(drop=True, inplace=True)

    return city



#returns mean value of Average Temperature

def city_mean_temp(city_name):

    """

    Returns the mean value of temperature for every year in a given city.

    """

    current_city = get_city_from_tempcsv(city_name)

    mean = current_city.groupby('year', as_index=False).mean()

    return mean



#returns mean value of Greenhouse gasses

def city_mean_gas(city_name):

    """

    Returns the mean value of Greenhouse gasses for every year in a given city.

    """

    current_city = get_city_from_gascsv(city_name)

    mean = current_city.groupby('year', as_index=False).mean()

    return mean



def plot_city_temps_with_spline(city_name):

    """

    Returns plot with spline based on temperature of a given city

    """

    data = city_mean_temp(city_name)

    x = data['year']

    y = data['AverageTemperature']

    

    x_new = np.linspace(1845, 2013, num=999, endpoint=True)

    spl = make_interp_spline(x, y, k=3)

    y_new = spl(x_new)

    plt.plot(x_new, y_new, colour[i], linewidth=4, label=city_name)



def plot_city_gas_with_spline(city_name):

    """

    Returns plot with spline based on gasses of a given city

    """

    data = city_mean_gas(city_name)

    x = data['year']

    y = data['Greenhouse']



    x_new = np.linspace(2014, 2020, num=7, endpoint=True)

    spl = make_interp_spline(x, y, k=3)

    y_new = spl(x_new)

    plt.plot(x_new, y_new, colour[i], linewidth=4, label=city_name)

    

    

    

#font dictionary

font = {'family': 'serif',

        'color':  'white',

        'weight': 'normal',

        'size': 25,

        }

    

    

#temperature graph

plt.figure(figsize=(20,10))



#labels for the temperature graph

plt.xlabel('Years', fontdict=font)

plt.ylabel('Average Temperature / Year', fontdict=font)

plt.title('Evolution Of Temperature In Time', fontdict=font)

ax=plt.gca()

ax.title.set_position([.5, 1.1])





#plotting evolution of temperature/year

for nume in cities:

    plot_city_temps_with_spline(nume)

    i = i + 1

plt.legend()

plt.grid(True)

plt.show()







#emissions graph

plt.figure(figsize=(20,10))



#labels for the Greenhouse gases graph

plt.xlabel('Years', fontdict=font)

plt.ylabel('Average Greenhouse Emissions / Year (tones)', fontdict=font)

plt.title('Evolution Of Greenhouse Gasses Emissions In Time', fontdict=font)

ax=plt.gca()

ax.title.set_position([.5, 1.1])





#plotting evolution of Greenhouse emissions/year

i = 0

nume = ""

for nume in cities:

    plot_city_gas_with_spline(nume)

    i = i + 1

plt.legend()

plt.show()