import numpy as np

import pandas as pd



# Import matplotlib.pyplot

import matplotlib.pyplot as plt



# Import seaborn library

import seaborn as sns

sns.set()



# Import plotly.plotly, 

# plotly.offline -> download_plotlyjs, init_notebook_mode, plot, iplot, and

# plotly.graph_objs

import chart_studio.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

import pycountry



import folium 

from folium import plugins



# Enable notebook mode

init_notebook_mode(connected = True)



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'



# To see the plots in the notebook

%matplotlib inline
raw_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

raw_data.head()
# See few data records

raw_data.head()
# Shape of the data

raw_data.shape
# Information about each columns

raw_data.info()
# Generates descriptive statistics

raw_data.describe()
# Checking missing values (column-wise)

raw_data.isnull().sum()
# Checking the percentage of missing values

round(100*(raw_data.isnull().sum()/len(raw_data.index)), 2)
# Dropping the rows with missing Province/State.

raw_data.dropna(inplace=True)
# Checking missing values (column-wise)

raw_data.isnull().sum()
raw_data["LastUpdated"] = pd.to_datetime(raw_data['Last Update'])
# Extract different components from the date



raw_data['date'] = pd.DatetimeIndex(raw_data['LastUpdated']).date



raw_data['year'] = pd.DatetimeIndex(raw_data['LastUpdated']).year



raw_data['month'] = pd.DatetimeIndex(raw_data['LastUpdated']).month



raw_data['day'] = pd.DatetimeIndex(raw_data['LastUpdated']).day



raw_data['time'] = pd.DatetimeIndex(raw_data['LastUpdated']).time



raw_data['dayofweek'] = pd.DatetimeIndex(raw_data['LastUpdated']).dayofweek



raw_data['day_name'] = pd.DatetimeIndex(raw_data['LastUpdated']).day_name()



raw_data['month_name'] = pd.DatetimeIndex(raw_data['LastUpdated']).month_name()

raw_data.head()
severity = (raw_data['Deaths'].sum() / raw_data['Confirmed'].sum())*100

severity
top_country = raw_data.groupby('Country').sum()

top_country['Country'] = top_country.index

top_country.sort_values(by='Confirmed', ascending=False).head(10)
countries = [country for country, df in raw_data.groupby('Country')]



plt.bar(countries, top_country['Confirmed'])

plt.xticks(countries, rotation='vertical', size=8)

plt.xlabel('Country name')

plt.ylabel('Number of Confirmed cases')

plt.show()
# Make a data frame with dots to show on the map

world_data = pd.DataFrame({

   'name':list(top_country['Country']),

    'lat':[-25.27,56.13,35.86,51.17,22.32,22.19,35.96,23.7,37.09],

   'lon':[133.78,-106.35,104.19,10.45,114.17,113.54,90.19,120.96,-95.71],

   'Confirmed':list(top_country['Confirmed']),

})



# create map and display it

world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='OpenStreetMap')



for lat, lon, value, name in zip(world_data['lat'], world_data['lon'], world_data['Confirmed'], world_data['name']):

    folium.CircleMarker([lat, lon],

                        radius=value * 0.001,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases </strong>: ' + str(value) + '<br>'),

                        color='purple',

                        

                        fill_color='indigo',

                        fill_opacity=0.7 ).add_to(world_map)



world_map
cntry = top_country['Country'].tolist()



confirmed = top_country['Confirmed'].tolist()



cntryCode = ['AUS','CAN', 'CHN','DEU','HKG','MAC','CHN', 'TWN', 'USA']



# Create a data using dict() method

data = dict(type = 'choropleth', # what type of plot you are doing

           locations = cntryCode, # list of abbreviated codes 

           #locationmode = 'USA-states', # locationmode for above abbreviated codes

           colorscale = 'Portland', # the colors you wanna plot

           text = cntry, # texts for the corresponding elements in locations parameter

           z = confirmed, # The color you want to represent for the corresponding elements in locations parameter

           colorbar = {'title' : 'Colorbar Title Goes Here'}) # Description about the color bar



layout = dict(title = 'Confirmed cases of Coronavirus',

              geo = dict(showframe = True,

                         showlakes = True, # Shows the actual lakes in the map

                     lakecolor = 'rgb(85, 173, 240)',

                         

                     projection = {'type' : 'equirectangular'}

                    ))



choromap = go.Figure(data = [data], layout = layout)



iplot(choromap)
mainland_china = raw_data.loc[raw_data['Country'] == 'Mainland China']



top_states = mainland_china.groupby('Province/State').sum()

top_states['Province/State'] = top_states.index

top_states.sort_values(by='Confirmed', ascending=False).head(10)
states = top_states['Province/State']



plt.bar(states, top_states['Confirmed'])

plt.xticks(states, rotation='vertical')

plt.xlabel('State name')

plt.ylabel('Number of Confirmed cases')

plt.show()
f, ax = plt.subplots(figsize=(20, 8))



sns.set_color_codes("pastel")

sns.barplot(x="Confirmed", y="Province/State", data=top_states,

            label="Confirmed", color="b")



sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Province/State", data=top_states,

            label="Recovered", color="g")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="upper right", frameon=True)

ax.set(xlim=(0, 2000), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)
# Create a pivot table on 'raw_data' dataset

fp = raw_data.pivot_table(index = 'Province/State', columns = 'Country', values = 'Confirmed')
# Plot the heatmap for the above pivot table

sns.heatmap(fp, cmap = 'plasma')
daily_confirmed = raw_data.groupby('date').sum()

daily_confirmed
dates = [date for date, df in raw_data.groupby('date')]

dates = pd.DatetimeIndex(dates).day



plt.plot(dates, daily_confirmed['Confirmed'])



plt.xticks(dates, rotation='vertical')

plt.show()
top_country['recovered_percent'] = (top_country['Recovered'] / top_country['Confirmed'])*100

top_country['death_percent'] = (top_country['Deaths'] / top_country['Confirmed'])*100



top_country.sort_values(by='recovered_percent', ascending=False).head(10)
# We can define the figure size while creating subplots: multiple subplots

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5), dpi = 100)



#ax.plot(dates, daily_confirmed['Confirmed'])

ax[0].plot(dates, daily_confirmed['Recovered'], label = 'Recovered')

ax[0].plot(dates, daily_confirmed['Deaths'], label = 'Deaths')



ax[1].plot(dates,daily_confirmed['Confirmed'], label = 'Confirmed')

ax[1].plot(dates,daily_confirmed['Recovered'], label = 'Recovered')

ax[1].plot(dates,daily_confirmed['Deaths'], label = 'Deaths')



plt.xticks(dates, rotation='vertical')

ax[0].legend()

ax[1].legend()

plt.tight_layout()

plt.show()
countries = top_country['Country']



# We can define the figure size while creating subplots: multiple subplots

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5), dpi = 100)



#ax.plot(dates, daily_confirmed['Confirmed'])

ax[0].plot(countries, top_country['Recovered'], label = 'Recovered')

ax[0].plot(countries, top_country['Deaths'], label = 'Deaths')

ax[0].set_xticklabels( countries, rotation=45);



ax[1].plot(countries,top_country['Confirmed'], label = 'Confirmed')

ax[1].plot(countries,top_country['Recovered'], label = 'Recovered')

ax[1].plot(countries,top_country['Deaths'], label = 'Deaths')

ax[1].set_xticklabels( countries, rotation=45);



ax[0].legend()

ax[1].legend()



plt.tight_layout()

plt.show()
plt.figure(figsize=(5,6))

#plt.subplot(1, 2, 1)

fig = top_country.boxplot(column='Confirmed')

fig.set_title('')

fig.set_ylabel('Confirmed')
countries = top_country['Country']



plt.bar(countries, top_country['Confirmed'])

plt.xticks(countries, rotation=45, size=8)

plt.xlabel('Country name')

plt.ylabel('Number of Confirmed cases')

plt.show()
plt.figure(figsize=(5,6))

#plt.subplot(1, 2, 1)

fig = top_states.boxplot(column='Confirmed')

fig.set_title('')

fig.set_ylabel('Confirmed')
states = top_states['Province/State']



plt.bar(states, top_states['Confirmed'])

plt.xticks(states, rotation='vertical')

plt.xlabel('State name')

plt.ylabel('Number of Confirmed cases')

plt.show()