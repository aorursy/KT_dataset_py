import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

import math

import seaborn as sns

import geopandas as gpd  # for geo plotting

import datetime as dt

import folium

from folium.plugins import HeatMap, MarkerCluster

import plotly.express as px



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
cov_deaths =  pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

cov_confirmed =  pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

cov_recoverd =  pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

cov_open_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

cov_papers = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

covid19 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

# covid19.head()
temp  =  covid19.groupby ('ObservationDate')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp = temp[temp.ObservationDate == temp.ObservationDate.max()]  #last day

temp['Global Mortality %'] = temp['Deaths']/temp['Confirmed']*100

temp['Global Recovery %'] = temp['Recovered']/temp['Confirmed']*100

temp.style.background_gradient(cmap='Pastel1')


for i in range(7):    

    x = dt.datetime.today() - dt.timedelta(days=i+2) # format the date to ddmmyyyy

    x = x.strftime ('%m/%d/%Y')

    temp2  =  covid19.groupby ('ObservationDate')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

    temp2 = temp2[temp2.ObservationDate == x]  #last day

    temp2['Global Mortality %'] = temp2['Deaths']/temp2['Confirmed']*100

    temp2['Global Recovery %'] = temp2['Recovered']/temp2['Confirmed']*100

    temp2.style.background_gradient(cmap='coolwarm')

#     print(temp2)

    print (temp2)
cntry  = covid19[covid19.ObservationDate == covid19.ObservationDate.max()].groupby ('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().sort_values(by='Confirmed', ascending=False)

cntry.style.background_gradient(cmap='Greys')
cntry  = covid19[covid19.ObservationDate == covid19.ObservationDate.max()].groupby ('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().sort_values(by='Deaths', ascending=False)

cntry.style.background_gradient(cmap='cividis')
prov  = covid19[covid19.ObservationDate == covid19.ObservationDate.max()].groupby (['Province/State', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].sum().sort_values(by='Deaths', ascending=False)

prov.head(20).style.background_gradient(cmap='terrain_r')
#segmentation by country

temp1 = covid19[covid19.ObservationDate == covid19.ObservationDate.max()]

temp1  =   temp1.groupby ('Country/Region')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()
df = temp1.sort_values(by= 'Deaths', ascending=False).head(20).sort_values(by= 'Deaths', ascending=True)

fig = px.bar(df, x= 'Deaths', y="Country/Region", orientation='h'  , text = 'Deaths' ,width=700, height=700, range_x = [0, max(temp1['Deaths'])+500])

fig.update_traces(marker_color='green', opacity=0.8, textposition='outside')

fig.update_layout(

    title={

        'text': "Top 20% Countries deaths cases",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
df = temp1.sort_values(by= 'Confirmed', ascending=False).head(20).sort_values(by= 'Confirmed', ascending=True)

fig = px.bar(df, x= 'Confirmed', y="Country/Region", orientation='h'  , text = 'Confirmed' , width=700, height=700, range_x = [0, max(temp1['Confirmed'])+2000])

fig.update_traces(marker_color='purple', opacity=0.8, textposition='outside')

fig.update_layout(

    title={

        'text': "Top 20% Countries Confirmed cases",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
df = temp1.sort_values(by= 'Recovered', ascending=False).head(20).sort_values(by= 'Recovered', ascending=True)

fig = px.bar(df, x= 'Recovered', y="Country/Region", orientation='h'  , text = 'Recovered' , width=700, height=700, range_x = [0, max(temp1['Recovered'])+2000])

fig.update_traces(marker_color='Orange', opacity=0.8, textposition='outside')

fig.update_layout(

    title={

        'text': "Top 20% Countries Recovered cases",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
tm = temp1.melt(id_vars="ObservationDate", value_vars=['Confirmed', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path=["variable"], values="value", height=400, width=600

                )

fig.show()
temp = covid19.groupby('ObservationDate')['Recovered', 'Deaths', 'Confirmed'].sum().reset_index()

temp = temp.melt(id_vars="ObservationDate", value_vars=['Recovered', 'Deaths', 'Confirmed'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="ObservationDate", y="Count", color='Case', height=800,

             title='Cases over time')

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
#new def cumul

new_data_df = covid19[["ObservationDate", "Province/State","Country/Region", "Deaths" ,"Confirmed" ,"Recovered"]].copy()

new_data_df.set_index('ObservationDate')

#create acumulatives columns 

# cumul_deaths = new_data_df.Deaths.cumsum()

# cumul_Confirmed = new_data_df.Confirmed.cumsum()

# cumul_Recovered = new_data_df.Recovered.cumsum()

# #adding columns

# new_data_df.insert(6, "cumul_deaths", cumul_deaths, True) 

# new_data_df.insert(7, "cumul_Confirmed", cumul_Confirmed, True) 

# new_data_df.insert(8, "cumul_Recovered", cumul_Recovered, True) 
with sns.axes_style('white'):

    g = sns.relplot(x="ObservationDate", y="Deaths" ,kind="line", data=new_data_df)

#     s = sns.relplot(x="Date", y="cumul_Recovered", kind="scatter", data=new_data_df)

    g.fig.autofmt_xdate()

    g.set_xticklabels(step=10)

    plt.title ("Covid-19 Deaths, Year:2020")

#     plt.legend()
with sns.axes_style('white'):

    g = sns.relplot(x="ObservationDate", y="Confirmed" ,kind="line", data=new_data_df)

    g.fig.autofmt_xdate()

    g.set_xticklabels(step=10)

    plt.title ("Covid-19 Confirmed, Year:2020")
with sns.axes_style('white'):

    g = sns.relplot(x="ObservationDate", y="Recovered" ,kind="line", data=new_data_df)

    g.fig.autofmt_xdate()

    g.set_xticklabels(step=10)

    plt.title ("Covid-19 Recovered, Year:2020")
temp = covid19.groupby(['ObservationDate', 'Country/Region'])['Confirmed'].sum().reset_index().sort_values('Confirmed', ascending=True)



fig = px.line(temp, x="ObservationDate", y="Confirmed", color='Country/Region', title='Confirmed Cases Spread by region', height=600)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()

#converting  lat and long to a format we can work with

cov = gpd.GeoDataFrame(cov_open_list, geometry=gpd.points_from_xy(cov_open_list["longitude"], cov_open_list["latitude"]))

cov.crs = {'init': 'epsg:4326'}
# Load a GeoDataFrame with country boundaries

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

ax = world.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')

cov.plot(ax=ax, markersize=10)
m = folium.Map(location=[0,0], zoom_start=2)

# filtered_data['R'] = np.sqrt(data_df['Recovered'])

HeatMap(data=cov_confirmed[['Lat', 'Long', cov_confirmed.columns[-1] ]].groupby(['Lat', 'Long']).sum().reset_index().values.tolist(),\

        radius=15, max_zoom=12).add_to(m)

m
m = folium.Map(location=[0,0], zoom_start=2)

# filtered_data['R'] = np.sqrt(data_df['Recovered'])

HeatMap(data=cov_deaths[['Lat', 'Long', cov_deaths.columns[-1] ]].groupby(['Lat', 'Long']).sum().reset_index().values.tolist(),\

        radius=15, max_zoom=12).add_to(m)

m
# m = folium.Map(location=[0,0], zoom_start=2)

# # filtered_data['R'] = np.sqrt(data_df['Recovered'])

# HeatMap(data=today_data[['Lat', 'Long', 'Recovered']].groupby(['Lat', 'Long']).sum().reset_index().values.tolist(),\

#         radius=15, max_zoom=12).add_to(m)

# m
# Confirmed

df = covid19[covid19.ObservationDate ==covid19.ObservationDate.max()].drop(['Province/State'], axis = 1)

fig = px.choropleth(df, locations="Country/Region", 

                    locationmode='country names', color=np.log(df["Confirmed"]), 

                    hover_name="Country/Region", hover_data=['Confirmed' , 'Recovered', 'Deaths'],

                    color_continuous_scale='Blues',                   

                    title='Countries with Confirmed Cases in the last day')

# fig.update(layout_coloraxis_showscale=False )

fig.show()
formated_gdf = covid19.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['ObservationDate'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 

                     color="Confirmed", size='size', hover_name= "Country/Region", 

                     range_color= [0, max(formated_gdf['Confirmed'])+2], 

                     projection="natural earth", animation_frame="Date", 

                     title='Spread over time confirmed cases')

fig.update(layout_coloraxis_showscale=False)

fig.show()
# Create a base map

m = folium.Map(location=[0,0], zoom_start=2)



mc = MarkerCluster()



for idx, row in cov_deaths.iterrows():

    if not math.isnan(row['Long']) and not math.isnan(row['Lat']):

        mc.add_child(folium.Marker([row['Lat'], row['Long']]))



m.add_child(mc)

fig = px.sunburst(cov_confirmed.sort_values(by=cov_confirmed.columns[-1], ascending=False).reset_index(drop=True), 

                 path=["Country/Region"], values= cov_confirmed.columns[-1], height=700,

                 title='Number of Confirmed cases reported'

                 )

fig.data[0].textinfo = 'label+text+value'

fig.show()
#tbd

temp = covid19.groupby(['ObservationDate','Country/Region'])['ObservationDate','Country/Region','Confirmed','Deaths','Recovered'].sum().reset_index()

temp = temp.sort_values(by = ['ObservationDate','Confirmed'] , ascending = True)

temp = temp.melt(id_vars=['Country/Region', 'ObservationDate'], value_vars=['Confirmed', 'Deaths', 'Recovered'], 

                 var_name='Case', value_name='Count').sort_values('Count')

fig = px.bar(temp, y='Country/Region', x= 'Count', height=500 ,barmode='group',

              title='World cases', animation_frame='ObservationDate',animation_group="Count",orientation='h',   range_x=[0, 70000])

fig.update_traces(textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.update_layout(yaxis={'categoryorder':'array',

                         'categoryarray':['Country/Region']})

fig.show()
sars_df = pd.read_csv('../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')

ebola_df = pd.read_csv('../input/ebola-outbreak-20142016-complete-dataset/ebola_2014_2016_clean.csv')

ebola_d =pd.read_csv('../input/ebola-outbreak-20142016-complete-dataset/ebola_data_db_format.csv')  #cumulative table per cases     
sars = sars_df.groupby('Date').sum().reset_index()

dd = sars.agg({'Date' : [np.min, np.max]}) #min & max date

sars =  sars[sars.Date == sars.Date.max()]

sars['Global Mortality %'] = sars['Number of deaths']/sars['Cumulative number of case(s)']*100

sars['Global Recovery %'] = sars['Number recovered']/sars['Cumulative number of case(s)']*100

sars.style.background_gradient(cmap='Reds')
dd.T
# ebola_df.head(5)
ebola_df.agg({'Date' : [np.min, np.max]}).T
ebola_df.Date =   pd.to_datetime(ebola_df.Date)
# print ('No of death: {}\nNo of cases: {}' .format(ebola_df[ebola_df['Date'] == ebola_df['Date'].max() ].sum()['No. of confirmed deaths'] ,ebola_df.count()['No. of confirmed cases'] ))
# ebola_df
ebola_df = ebola_df[ebola_df['Date'] == ebola_df['Date'].max() ]

ebola_df = ebola_df[['Country', 'No. of confirmed cases','No. of confirmed deaths']]

ebola_df  = ebola_df.groupby('Country').sum().sort_values(by = 'No. of confirmed cases', ascending = False).reset_index()

ebola_df['Global Mortality %'] = ebola_df['No. of confirmed deaths']/ebola_df['No. of confirmed cases']*100

print ('No of death: {}\nNo of cases: {}' .format(ebola_df['No. of confirmed deaths'].sum() ,ebola_df['No. of confirmed cases'].sum() ))
fig = px.sunburst(ebola_df.sort_values(by='Global Mortality %', ascending=False).reset_index(drop=True), 

                 path=["Country"], values= 'Global Mortality %', height=500,

                 title='Percentage of Countries contracted by Ebola'

                 )

fig.data[0].textinfo = 'label+text+value'

fig.show()
df = ebola_df.sort_values(by= 'No. of confirmed deaths', ascending=False).head(5).sort_values(by= 'No. of confirmed deaths', ascending=True)

fig = px.bar(df, x= 'No. of confirmed deaths', y="Country", orientation='h'  , text = 'No. of confirmed deaths' , width=700, height=700, range_x = [0,4000])

fig.update_traces(marker_color='olivedrab', opacity=0.8, textposition='outside')

fig.update_layout(

    title={

        'text': "Top Confirmed deaths countries",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
df = ebola_df.sort_values(by= 'No. of confirmed cases', ascending=False).head(20).sort_values(by= 'No. of confirmed cases', ascending=True)

fig = px.bar(df, x= 'No. of confirmed cases', y="Country", orientation='h'  , text = 'No. of confirmed cases' , width=700, height=700, range_x = [0,9000])

fig.update_traces(marker_color='crimson', opacity=0.8, textposition='outside')

fig.update_layout(

    title={

        'text': "Top Confirmed countries",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()