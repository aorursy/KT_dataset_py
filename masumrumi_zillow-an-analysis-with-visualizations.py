# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

from IPython.display import Image, display



# import plotly

import plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.tools as tls

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as fig_fact

plotly.tools.set_config_file(world_readable=True, sharing='public')





%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
state_time_series = pd.read_csv('../input/State_time_series.csv')

city_time_series = pd.read_csv('../input/City_time_series.csv')
"""## Importing all the datasets. Since this kernel will be continous working progress. 

cities_crosswalk = pd.read_csv('../input/cities_crosswalk.csv')



county_time_series = pd.read_csv('../input/County_time_series.csv')

metro_time_series = pd.read_csv('../input/Metro_time_series.csv')

neighborhood_time_series = pd.read_csv('../input/Neighborhood_time_series.csv')

zip_time_series = pd.read_csv('../input/Zip_time_series.csv')

countyCrossWalk_Zillow = pd.read_csv('../input/CountyCrossWalk_Zillow.csv')"""
state_time_series.head(2)
## Finding out top ten states with most ZHVI values. 

f, ax = plt.subplots(2,1, figsize=(20, 30))

state_time_series.Date = pd.to_datetime(state_time_series.Date) ## 

state_without_null = state_time_series.dropna(subset=['ZHVI_AllHomes'], how='any') ## getting rid of null values

state_without_null.Date = state_without_null.Date.dt.year ## set date to year.



state_without_null.groupby(state_without_null.RegionName)['ZHVI_AllHomes'].mean().sort_values().plot( kind = 'bar', ax =ax[0], color = 'g')

ax[0].set_title('States with Most Home Values According to Zillow', fontsize = 25)

ax[0].set_xlabel('States')

ax[0].set_ylabel('Price')





state_without_null.groupby(state_without_null.RegionName)['ZHVI_AllHomes'].mean().sort_values().tail(10).plot(kind = 'bar',ax =ax[1], color = 'g')

ax[1].set_title('Top States with Most Home Values According to Zillow', fontsize = 25)

ax[1].set_xlabel('States')

ax[1].set_ylabel('Price')
## lets create a dataframe with only the top states

top_states = state_without_null[(state_without_null.RegionName == "NewYork")|

                   (state_without_null.RegionName == "Washington")|

                   (state_without_null.RegionName == "Colorado")|

                   (state_without_null.RegionName == "Connecticut")|

                   (state_without_null.RegionName == "Maryland")|

                   (state_without_null.RegionName == "NewJersey")|

                   (state_without_null.RegionName == "Alaska")|

                   (state_without_null.RegionName == "Massachusetts")|

                   (state_without_null.RegionName == "California")|

                   (state_without_null.RegionName == "Hawaii")|

                   (state_without_null.RegionName == "DistrictofColumbia")]
pd.pivot_table(top_states, 

               index="Date", 

               columns="RegionName", 

               values="ZHVI_AllHomes").plot(kind = 'line', figsize = (20,10),colormap='flag', legend = True)

plt.xlabel("Year", fontsize = 15)

plt.ylabel("$ Price",  fontsize = 15)

plt.title(" ZHVI home values change in overtime for top expensive states", fontsize = 24)
## Lets see how the rent is for these states. 

state_time_series.Date = pd.to_datetime(state_time_series.Date)

state_without_null = state_time_series.dropna(subset=['ZRI_AllHomes'], how='any')

state_without_null.Date = state_without_null.Date.dt.year



plt.subplots(figsize = (14,6))

state_without_null.groupby(state_without_null.RegionName)['ZRI_AllHomes'].mean().sort_values().tail(10).plot(kind = 'bar')

plt.title('Top States with Most Rent Values According to Zillow', fontsize = 25)

plt.xlabel('States')

plt.ylabel('dollar amount');
states = {

        'AK': 'Alaska',

        'AL': 'Alabama',

        'AR': 'Arkansas',

        'AS': 'American Samoa',

        'AZ': 'Arizona',

        'CA': 'California',

        'CO': 'Colorado',

        'CT': 'Connecticut',

        'DC': 'DistrictofColumbia',

        'DE': 'Delaware',

        'FL': 'Florida',

        'GA': 'Georgia',

        'GU': 'Guam',

        'HI': 'Hawaii',

        'IA': 'Iowa',

        'ID': 'Idaho',

        'IL': 'Illinois',

        'IN': 'Indiana',

        'KS': 'Kansas',

        'KY': 'Kentucky',

        'LA': 'Louisiana',

        'MA': 'Massachusetts',

        'MD': 'Maryland',

        'ME': 'Maine',

        'MI': 'Michigan',

        'MN': 'Minnesota',

        'MO': 'Missouri',

        'MP': 'Northern Mariana Islands',

        'MS': 'Mississippi',

        'MT': 'Montana',

        'NA': 'National',

        'NC': 'NorthCarolina',

        'ND': 'NorthDakota',

        'NE': 'Nebraska',

        'NH': 'NewHampshire',

        'NJ': 'NewJersey',

        'NM': 'NewMexico',

        'NV': 'Nevada',

        'NY': 'NewYork',

        'OH': 'Ohio',

        'OK': 'Oklahoma',

        'OR': 'Oregon',

        'PA': 'Pennsylvania',

        'PR': 'Puerto Rico',

        'RI': 'RhodeIsland',

        'SC': 'SouthCarolina',

        'SD': 'SouthDakota',

        'TN': 'Tennessee',

        'TX': 'Texas',

        'UT': 'Utah',

        'VA': 'Virginia',

        'VI': 'VirginIslands',

        'VT': 'Vermont',

        'WA': 'Washington',

        'WI': 'Wisconsin',

        'WV': 'WestVirginia',

        'WY': 'Wyoming'

}



states_w_n_zhvi_all = state_without_null.groupby(state_without_null.RegionName)['ZHVI_AllHomes'].mean().reset_index(name = 'ZHVI_AllHomes')

states_w_n_zhvi_all.dropna(inplace=True)
states_w_n_zhvi_all.head(2)
## Replacing the state names into acronyms. 

## This list comprehension is broken down in order to have a better understanding of the code

states_w_n_zhvi_all.RegionName = [

    j ## output

    for j in states ## outer loop 

    for i in states_w_n_zhvi_all["RegionName"] ## inner loop 

    if i == states[j]] ## if statement
states_w_n_zhvi_all.head(2)
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],[0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]





# difine our data for plotting

data = [ dict(

        type='choropleth',

        colorscale = scl,

        autocolorscale = True,

        locations = states_w_n_zhvi_all.RegionName, # location (states)

        z = states_w_n_zhvi_all['ZHVI_AllHomes'].astype(float), # Zillow Home value per square foot

        locationmode = 'USA-states', # let's define the location mode to USA_states

        text = "Average zillow home value",

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(

            title = "Mean ZHVI for all home")

        ) ]



layout = dict(

        title = 'Median of the value of all homes per square foot in different states<br>(Hover for breakdown)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(0, 200, 255)'),

             )









    

fig = dict( data=data, layout=layout )

# let's plot

py.iplot( fig, filename='d3ropleth-map' )
city_time_series.Date = pd.to_datetime(city_time_series.Date)

zhvi_sqf_all = city_time_series.dropna(subset=['ZHVIPerSqft_AllHomes'],how='any')
plt.subplots(figsize = (14,8))

zhvi_sqf_all.groupby(zhvi_sqf_all.Date.dt.year)['ZHVIPerSqft_AllHomes'].mean().plot(kind = 'bar', width = .95, color = 'g')

plt.ylabel("price per sqft")

plt.xlabel("Year")

plt.title("Changes in all home prices per sqft overtime", fontsize = 24)

plt.show()



zri_all = city_time_series.dropna(subset=['MedianRentalPrice_AllHomes'], how='any');

fig, ax = plt.subplots(figsize = (14,8))

zri_all.groupby(zri_all.Date.dt.year)['MedianRentalPrice_AllHomes'].mean().plot(kind = 'bar',width = .95, color = 'g')
zri_all = city_time_series.dropna(subset=['MedianListingPrice_AllHomes'], how='any');

fig, ax = plt.subplots(figsize = (14,8))

zri_all.groupby(zri_all.Date.dt.year)['MedianListingPrice_AllHomes'].mean().plot(kind = 'bar',width = .95, color = 'g')