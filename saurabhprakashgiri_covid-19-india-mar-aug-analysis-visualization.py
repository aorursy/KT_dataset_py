import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
district_data = pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/district_level_latest.csv")

state_data = pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/state_level_latest.csv")

country_data = pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/nation_level_daily.csv")

patient_data = pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/patients_data.csv")

tests_state_data = pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/tests_state_wise.csv")
district_data = district_data.drop(['District_Notes','Last_Updated'],axis=1)

district_data.info()
# Drop the column of Totals

state_data = state_data.drop(0)



# Plot the Confirmed, Recovered, Active and Deaths on the same plot using matplotlib

plt.figure(figsize=(12,10))

x = state_data['State_code']

y = state_data['Confirmed']

plt.plot(x,y,marker='o',label="Confirmed")

y = state_data['Recovered']

plt.plot(x,y,marker='o',label="Recovered")

y = state_data['Active']

plt.plot(x,y,marker='o',label="Active")

y = state_data['Deaths']

plt.plot(x,y,marker='o',label="Deaths")

plt.legend();
state_top20 = state_data.nlargest(20,'Confirmed')

state_top20
# Position of bars on x-axis

ind = np.arange(20)



# Width of a bar 

width = 0.4



plt.figure(figsize=(15,12))

x = state_top20['State_code']

y = state_top20['Confirmed']

plt.bar(ind+width/2,y,align='edge',width=width,label="Confirmed")

y = state_top20['Recovered']

plt.bar(ind+width,y,align='edge',width=width,label="Recovered")

y = state_top20['Active']

plt.bar(ind+3*width/2,y,align='edge',width=width,label="Active")

y = state_top20['Deaths']

plt.bar(ind+2*width,y,align='edge',width=width,label="Deaths")



plt.xticks(ind + 3*width/2, x)

plt.legend();

district_top10 = district_data.nlargest(10,'Confirmed')

district_top10
# Position of bars on x-axis

ind = np.arange(10)



# Width of a bar 

width = 0.4



plt.figure(figsize=(15,12))

x = district_top10['District']

y = district_top10['Confirmed']

plt.bar(ind+width/2,y,align='edge',width=width,label="Confirmed")

y = district_top10['Recovered']

plt.bar(ind+width,y,align='edge',width=width,label="Recovered")

y = district_top10['Active']

plt.bar(ind+3*width/2,y,align='edge',width=width,label="Active")

y = district_top10['Deceased']

plt.bar(ind+2*width,y,align='edge',width=width,label="Deseased")



plt.xticks(ind + 3*width/2, x)

plt.legend()
tests = tests_state_data.loc[tests_state_data['Updated On'] == '06/08/2020']

tests = tests[['State','Total Tested','Positive']]

tests = tests.dropna()

tests.info()
import plotly.express as px



fig = px.scatter(tests, x="Total Tested", y="Positive", text="State", log_x=True, log_y=True, size_max=100, color="Positive")

fig.update_traces(textposition='top center')

fig.update_layout(title_text='Life Expectency', title_x=0.5)

fig.show() 

# Hover over the image to see the details of each state
import geopandas as gpd



fp = r'../input/indiageofiles/india-polygon.shp'

map_df = gpd.read_file(fp)

map_df.rename(columns={'st_nm': 'State'},inplace=True)

map_df.head() #check the head of the file
data_merge = map_df.merge(state_data, on = 'State', how = 'left')

data_merge.head()
fig, ax = plt.subplots(1, figsize=(10, 10))

ax.axis('off')

ax.set_title('Statewise Confirmed Cases', fontdict={'fontsize': '25', 'fontweight' : '10'})



# plot the figure

data_merge.plot(column='Confirmed',cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0', legend=True,markersize=[39.739192, -104.990337]);
fig, ax = plt.subplots(1, figsize=(10, 10))

ax.axis('off')

ax.set_title('Statewise Recovered Cases', fontdict={'fontsize': '25', 'fontweight' : '10'})

data_merge.plot(column='Recovered',cmap='Greens', linewidth=0.8, ax=ax, edgecolor='0', legend=True,markersize=[39.739192, -104.990337]);

fig, ax = plt.subplots(1, figsize=(10, 10))

ax.axis('off')

ax.set_title('Statewise Deaths', fontdict={'fontsize': '25', 'fontweight' : '10'})

data_merge.plot(column='Deaths',cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0', legend=True,markersize=[39.739192, -104.990337]);
import branca.colormap as cm

import folium

states_daily_data = pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/state_level_daily.csv")

states_daily_data.head()
states =states_daily_data.drop(['State'],axis=1)

states = states.rename(columns={'State_Name': 'State'})

states_map = states.merge(map_df,on='State')

states_map.head()
states_df = states_map[states_map.Confirmed != 0]

states_df.info()
states_df['log_Confirmed'] = np.log10(states_df['Confirmed'])
states_df.head()
states_df['date'] = pd.to_datetime(states_df['Date']).astype(int) / 10**9

states_df['date'] = states_df['date'].astype(int).astype(str)
states_df.head()
states_df = states_df[['State', 'date', 'log_Confirmed', 'geometry']]

states_df = states_df.dropna() #Drop all the rows with any null values

states_df.info()
states_df = states_df.sort_values(['State','date']).reset_index(drop=True)

states_df.head()
max_colour = max(states_df['log_Confirmed'])

min_colour = min(states_df['log_Confirmed'])

cmap = cm.linear.YlOrRd_09.scale(min_colour, max_colour)

states_df['colour'] = states_df['log_Confirmed'].map(cmap)
states_list = states_df['State'].unique().tolist()

states_idx = range(len(states_list))



style_dict = {}

for i in states_idx:

    states = states_list[i]

    result = states_df[states_df['State'] == states]

    inner_dict = {}

    for _, r in result.iterrows():

        inner_dict[r['date']] = {'color': r['colour'], 'opacity': 0.7}

    style_dict[str(i)] = inner_dict
states_geom_df = states_df[['geometry']]

states_geom_gdf = gpd.GeoDataFrame(states_geom_df)

states_geom_gdf = states_geom_gdf.drop_duplicates().reset_index()
from folium.plugins import TimeSliderChoropleth



slider_map = folium.Map(zoom_start=4,location=[21, 78])



_ = TimeSliderChoropleth(

    data=states_geom_gdf.to_json(),

    styledict=style_dict,



).add_to(slider_map)



_ = cmap.add_to(slider_map)

cmap.caption = "Log of number of confirmed cases"

slider_map.save(outfile='Covid19_Map_India.html')

from IPython.display import IFrame

IFrame(src='./Covid19_Map_India.html', width=700, height=600)



#Slide the bar to see the Covid-19 spread across Indian states from 14 March'20 to 6 August'20