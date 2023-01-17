# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib as plt
import folium 
from folium import plugins
from fbprophet import Prophet
import plotly.offline as py
# set up 

dataframe = pd.read_csv("/kaggle/input/covid19-us-county-jhu-data-demographics/covid_us_county.csv")
df = dataframe

df.describe()

df.tail(50000)
#Settings for the prediction :)

# you can choose how far in the future should the prediction go and how much high the confidency interval should be

#Amount on months: (1 Month = 30,2 Months = 60, 3 Months = 90)
mo = 60

#Confidence interval (90% = 0.9, 95% = 0.95, 99% = 0.99)
con = 0.95
#preparing colnames for cases for prediction- group the data by date and sum up cases (also do it for every of the top 5 states)
cases_CA = df.query('state=="California"').groupby('date')[['cases']].sum().reset_index()
cases_CA = cases_CA.rename(columns={"date": "ds", "cases": "y"})

cases_PE = df.query('state=="Pennsylvania"').groupby('date')[['cases']].sum().reset_index()
cases_PE = cases_PE.rename(columns={"date": "ds", "cases": "y"})

cases_MA = df.query('state=="Massachusetts"').groupby('date')[['cases']].sum().reset_index()
cases_MA = cases_MA.rename(columns={"date": "ds", "cases": "y"})

cases_NY = df.query('state=="New York"').groupby('date')[['cases']].sum().reset_index()
cases_NY = cases_NY.rename(columns={"date": "ds", "cases": "y"})

cases_IL = df.query('state=="Illinois"').groupby('date')[['cases']].sum().reset_index()
cases_IL = cases_IL.rename(columns={"date": "ds", "cases": "y"})
#preparing colnames for deaths for prediction same as above just for deaths
deaths_CA = df.query('state=="California"').groupby('date')[['deaths']].sum().reset_index()
deaths_CA = deaths_CA.rename(columns={"date": "ds", "deaths": "y"})
deaths_CA['ds'] = pd.to_datetime(deaths_CA['ds'])

deaths_PE = df.query('state=="Pennsylvania"').groupby('date')[['deaths']].sum().reset_index()
deaths_PE = deaths_PE.rename(columns={"date": "ds", "deaths": "y"})
deaths_PE['ds'] = pd.to_datetime(deaths_PE['ds'])

deaths_MA = df.query('state=="Massachusetts"').groupby('date')[['deaths']].sum().reset_index()
deaths_MA = deaths_MA.rename(columns={"date": "ds", "deaths": "y"})
deaths_MA['ds'] = pd.to_datetime(deaths_MA['ds'])

deaths_NY = df.query('state=="New York"').groupby('date')[['deaths']].sum().reset_index()
deaths_NY = deaths_NY.rename(columns={"date": "ds", "deaths": "y"})
deaths_NY['ds'] = pd.to_datetime(deaths_NY['ds'])

deaths_IL = df.query('state=="Illinois"').groupby('date')[['deaths']].sum().reset_index()
deaths_IL = deaths_IL.rename(columns={"date": "ds", "deaths": "y"})
deaths_IL['ds'] = pd.to_datetime(deaths_IL['ds'])
#Model for CA
mc_CA = Prophet(interval_width=con)
mc_CA.fit(cases_CA)
future = mc_CA.make_future_dataframe(periods=mo,include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_CA = mc_CA.predict(future)
forecast_CA[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_CA = mc_CA.plot(forecast_CA)
#Model for CA Deaths
md_CA = Prophet(interval_width=con)
md_CA.fit(deaths_CA)
future_CAD = md_CA.make_future_dataframe(periods=mo, include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_CAD = md_CA.predict(future_CAD)
forecast_CAD[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_CAD = md_CA.plot(forecast_CAD)
#Model for PE
mc_PE = Prophet(interval_width=con)
mc_PE.fit(cases_PE)
future = mc_PE.make_future_dataframe(periods=mo,include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_PE = mc_PE.predict(future)
forecast_PE[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_PE = mc_PE.plot(forecast_PE)
#Model for PE Deaths
md_PE = Prophet(interval_width=con)
md_PE.fit(deaths_PE)
future_PED = md_PE.make_future_dataframe(periods=mo, include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_PED = md_PE.predict(future_PED)
forecast_PED[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_PED = md_PE.plot(forecast_PED)
#Model for IL
mc_IL = Prophet(interval_width=con)
mc_IL.fit(cases_IL)
future = mc_IL.make_future_dataframe(periods=mo,include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_IL = mc_IL.predict(future)
forecast_IL[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_IL = mc_IL.plot(forecast_IL)
#Model for IL Deaths
md_IL = Prophet(interval_width=con)
md_IL.fit(deaths_IL)
future_ILD = md_IL.make_future_dataframe(periods=mo, include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_ILD = md_PE.predict(future_ILD)
forecast_ILD[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_ILD = md_IL.plot(forecast_ILD)
#Model for MA
mc_MA = Prophet(interval_width=con)
mc_MA.fit(cases_MA)
future = mc_MA.make_future_dataframe(periods=mo,include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_MA = mc_MA.predict(future)
forecast_MA[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_MA = mc_MA.plot(forecast_MA)
#Model for MA Deaths
md_MA = Prophet(interval_width=con)
md_MA.fit(deaths_MA)
future_MAD = md_MA.make_future_dataframe(periods=mo, include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_MAD = md_MA.predict(future_MAD)
forecast_MAD[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_MAD = md_MA.plot(forecast_MAD)
#Model for NY Cases
mc_NY = Prophet(interval_width=con)
mc_NY.fit(cases_NY)
future = mc_NY.make_future_dataframe(periods=mo, include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_NY = mc_NY.predict(future)
forecast_NY[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_NY = mc_NY.plot(forecast_NY)
#Model for NY Deaths
md_NY = Prophet(interval_width=con)
md_NY.fit(deaths_NY)
future_D = md_NY.make_future_dataframe(periods=mo, include_history=True) #prediction period 1 month = 30 days
#future.tail()

forecast_NYD = md_NY.predict(future_D)
forecast_NYD[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_NY = md_NY.plot(forecast_NYD)
com_NY = md_NY.plot(forecast_NYD), mc_NY.plot(forecast_NY)
com_PE = md_PE.plot(forecast_PED), mc_PE.plot(forecast_PE)
com_MA = md_MA.plot(forecast_MAD), mc_MA.plot(forecast_MA)
com_IL = md_IL.plot(forecast_ILD), mc_IL.plot(forecast_IL)
com_CA = md_CA.plot(forecast_CAD), mc_CA.plot(forecast_CA)
#preparing colnames for cases for prediction test same as preparation in the regular ones just for test/train data
cases_CA_test = df.query('state=="California"').groupby('date')[['cases']].sum().reset_index() #group cases from california by date and sum cases
cases_CA_test = cases_CA_test.rename(columns={"date": "ds", "cases": "y"})
cases_CA_test = cases_CA_test[:-30] #drop last 30 days

cases_PE_test = df.query('state=="Pennsylvania"').groupby('date')[['cases']].sum().reset_index()
cases_PE_test = cases_PE_test.rename(columns={"date": "ds", "cases": "y"})
cases_PE_test = cases_PE_test[:-30]

cases_MA_test = df.query('state=="Massachusetts"').groupby('date')[['cases']].sum().reset_index()
cases_MA_test = cases_MA_test.rename(columns={"date": "ds", "cases": "y"})
cases_MA_test = cases_MA_test[:-30]

cases_NY_test = df.query('state=="New York"').groupby('date')[['cases']].sum().reset_index()
cases_NY_test = cases_NY_test.rename(columns={"date": "ds", "cases": "y"})
cases_NY_test = cases_NY_test[:-30]

cases_IL_test = df.query('state=="Illinois"').groupby('date')[['cases']].sum().reset_index()
cases_IL_test = cases_IL_test.rename(columns={"date": "ds", "cases": "y"})
cases_IL_test = cases_IL_test[:-30]

cases_IL_test.tail(10)
#preparing colnames for deaths test for prediction same as preparation in the regular ones just for test/train data
deaths_CA_test = df.query('state=="California"').groupby('date')[['deaths']].sum().reset_index()
deaths_CA_test = deaths_CA_test.rename(columns={"date": "ds", "deaths": "y"})
deaths_CA_test['ds'] = pd.to_datetime(deaths_CA_test['ds'])
deaths_CA_test = deaths_CA_test[:-30]

deaths_PE_test = df.query('state=="Pennsylvania"').groupby('date')[['deaths']].sum().reset_index()
deaths_PE_test = deaths_PE_test.rename(columns={"date": "ds", "deaths": "y"})
deaths_PE_test['ds'] = pd.to_datetime(deaths_PE_test['ds'])
deaths_PE_test = deaths_PE_test[:-30]

deaths_MA_test = df.query('state=="Massachusetts"').groupby('date')[['deaths']].sum().reset_index()
deaths_MA_test = deaths_MA_test.rename(columns={"date": "ds", "deaths": "y"})
deaths_MA_test['ds'] = pd.to_datetime(deaths_MA_test['ds'])
deaths_MA_test =deaths_MA_test[:-30]

deaths_NY_test = df.query('state=="New York"').groupby('date')[['deaths']].sum().reset_index()
deaths_NY_test = deaths_NY_test.rename(columns={"date": "ds", "deaths": "y"})
deaths_NY_test['ds'] = pd.to_datetime(deaths_NY_test['ds'])
deaths_NY_test = deaths_NY_test[:-30]

deaths_IL_test = df.query('state=="Illinois"').groupby('date')[['deaths']].sum().reset_index()
deaths_IL_test = deaths_IL_test.rename(columns={"date": "ds", "deaths": "y"})
deaths_IL_test['ds'] = pd.to_datetime(deaths_IL_test['ds'])
deaths_IL_test = deaths_IL_test[:-30]
Cases
#Model for NY Test Cases
mc_NY_test = Prophet(interval_width=con)
mc_NY_test.fit(cases_NY_test)
future = mc_NY_test.make_future_dataframe(periods=30, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_NY_test = mc_NY_test.predict(future)
forecast_NY_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_NY = mc_NY_test.plot(forecast_NY_test)
#Model for NY Test Cases
md_NY_test = Prophet(interval_width=con)
md_NY_test.fit(deaths_NY_test)
future = md_NY_test.make_future_dataframe(periods=30, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_NY_test_d = md_NY_test.predict(future)
forecast_NY_test_d[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

deaths_forecast_plot_NY = md_NY_test.plot(forecast_NY_test_d)
#Model for PE Test Cases
mc_PE_test = Prophet(interval_width=con)
mc_PE_test.fit(cases_PE_test)
future = mc_PE_test.make_future_dataframe(periods=mo, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_PE_test = mc_PE_test.predict(future)
forecast_PE_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_PE = mc_PE_test.plot(forecast_PE_test)
#Model for PE Test Cases
md_PE_test = Prophet(interval_width=con)
md_PE_test.fit(deaths_PE_test)
future = md_PE_test.make_future_dataframe(periods=mo, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_PE_test_d = md_PE_test.predict(future)
forecast_PE_test_d[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

deaths_forecast_plot_PE = md_PE_test.plot(forecast_PE_test_d)
#Model for IL Test Cases - I don't know what's the issue here guys
mc_IL_test = Prophet(interval_width=con)
mc_IL_test.fit(cases_IL_test)
future = mc_IL_test.make_future_dataframe(periods=mo, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_IL_test = mc_IL_test.predict(future)
forecast_IL_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_IL_test = mc_IL_test.plot(forecast_IL_test)
#Model for IL Test Cases
md_IL_test = Prophet(interval_width=con)
md_IL_test.fit(deaths_IL_test)
future = md_IL_test.make_future_dataframe(periods=mo, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_IL_test_d = md_IL_test.predict(future)
forecast_IL_test_d[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

deaths_forecast_plot_IL = md_IL_test.plot(forecast_IL_test_d)
#Model for MA Test Cases
mc_MA_test = Prophet(interval_width=con)
mc_MA_test.fit(cases_MA_test)
future = mc_MA_test.make_future_dataframe(periods=mo, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_MA_test = mc_MA_test.predict(future)
forecast_MA_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_MA = mc_MA_test.plot(forecast_MA_test)
#Model for MA Test Cases
md_MA_test = Prophet(interval_width=con)
md_MA_test.fit(deaths_MA_test)
future = md_MA_test.make_future_dataframe(periods=mo, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_MA_test_d = md_MA_test.predict(future)
forecast_MA_test_d[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

deaths_forecast_plot_MA = md_MA_test.plot(forecast_MA_test_d)
#Model for CA Test Cases
mc_CA_test = Prophet(interval_width=con)
mc_CA_test.fit(cases_MA_test)
future = mc_CA_test.make_future_dataframe(periods=mo, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_CA_test = mc_CA_test.predict(future)
forecast_CA_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

cases_forecast_plot_CA = mc_CA_test.plot(forecast_CA_test)
#Model for CA Test Cases
md_CA_test = Prophet(interval_width=con)
md_CA_test.fit(deaths_CA_test)
future = md_CA_test.make_future_dataframe(periods=mo, include_history=True) #prediction time based on variable mo
#future.tail()

forecast_CA_test_d = md_CA_test.predict(future)
forecast_CA_test_d[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

deaths_forecast_plot_CA = md_CA_test.plot(forecast_CA_test_d)
us_covid = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/covid_us_county.csv')
us_county = pd.read_csv('../input/covid19-us-county-jhu-data-demographics/us_county.csv')
# set na as object
us_covid = us_covid[us_covid.fips.notnull()]
us_covid['fips'] = us_covid['fips'].astype(object)
us_county['fips'] = us_county['fips'].astype(object)
#add the column popultation from the county data
us_covid = us_covid.merge(us_county[['fips', 'population']], on = ['fips'], how = "left")

#cut out every date but the last one, with sorting the dataset and then keeping the latest
us_cum = us_covid.sort_values(by = ['county', 'state', 'date'], ascending = [True, True, False]) #latest date is on top
us_cum = us_cum.drop_duplicates(subset = ['county', 'state'], keep = "first")                    #cut out every date older than the latest
#save a copy
#counties_us = us_cum.copy()
#group by state for state view 
us_cum = us_cum.groupby(['state', 'date'], as_index=False)['cases', 'deaths', 'population'].sum()

us_cum['population'] = us_cum['population'].astype(int)
#rounding population to millions with 2 digits, and creating two new columns called cases per million and deaths per million
us_cum['population'] = round((us_cum['population']/1000000),2)
us_cum = us_cum.rename(columns = {'population': 'Population (million)'})
us_cum['Cases per Million'] = round((us_cum['cases']/us_cum['Population (million)']),2)
us_cum['Deaths per Million'] = round((us_cum['deaths']/us_cum['Population (million)']),2)

url = '../input/usa-states'         # adress of the states file in the input folder
state_geo = f'{url}/usa-states.json' #import the pyligons from the usa states dataset (json file)

bins = list(us_cum['Cases per Million'].quantile([0, 0.5, 0.75, 0.90, 0.95, 1])) # set the steps for the legend

map1 = folium.Map(location=[34, -118], zoom_start=4)   #creat map and set zoom step and location (autofocus on californias longitude and altitude)

choropleth = folium.Choropleth(                        #setting for the graph
    geo_data=state_geo,
    name='choropleth',
    data=us_cum,
    columns=['state', 'Cases per Million'],
    key_on='properties.name',
    fill_color= 'PuBuGn',
    fill_opacity=0.75,
    line_opacity=0.2,
    legend_name='Cases per Million',
    bins = bins,
    reset = True
).add_to(map1)

style_function = "font-size: 15px; font-weight: bold"
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['name'],style=style_function, labels=False)     #mouse over feature for displaying the state names
)

map1   # show the map
url = '../input/usa-states'          #functions like the graph above just with deaths
state_geo = f'{url}/usa-states.json'
map2 = folium.Map(location=[34, -118], zoom_start=4)

bins = list(us_cum['Deaths per Million'].quantile([0, 0.5, 0.75, 0.90, 0.95, 1]))

choropleth = folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=us_cum,
    columns=['state', 'Deaths per Million'],
    key_on='properties.name',
    fill_color= 'PuBuGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Deaths per Million',
    bins = bins,
    reset = True
).add_to(map2)

style_function = "font-size: 15px; font-weight: bold"
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['name'],style=style_function, labels=False)
)

map2
map3 = folium.Map(location=[34, -118], zoom_start=4) #functions like the graph above just with absoulte number of cases

choropleth = folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=us_cum,
    columns=['state', 'cases'],
    key_on='properties.name',
    fill_color= 'PuBuGn',
    fill_opacity=0.75,
    line_opacity=0.2,
    legend_name='cases',
    reset = True
).add_to(map3)

style_function = "font-size: 15px; font-weight: bold"
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['name'],style=style_function, labels=False)
)

map3
############## Trashbin #####################
#############################################
#############################################
#############################################
#############################################
#############################################
#State selection: nope
#CA - California
#PE - Pensylvania
#MA - Massachusetts
#NY - New York
#IL - Illinios
#i = pd.DataFrame(
#    {"code": ['CA','PE','MA','NY','IL'],
#     "name": ['California','Pensylvania','Massachusetts','New York','Illinois']},
#    index = [1,2,3,4,5])

#print(i)
#creating subsets for top 5 states for descriptive stuff:

#df_CA = df.query('state=="California"').groupby('date')[['cases','deaths']].sum().reset_index()
#df_PE = df.query('state=="Pensylvania"').groupby('date')[['cases','deaths']].sum().reset_index()
#df_MA = df.query('state=="Massachusetts"').groupby('date')[['cases','deaths']].sum().reset_index()
#df_NY = df.query('state=="New York"').groupby('date')[['cases','deaths']].sum().reset_index()
#df_IL = df.query('state=="Illinois"').groupby('date')[['cases','deaths']].sum().reset_index()

#ds_PE = df.query('state=="Pensylvania"').reset_index()
#ds_MA = df.query('state=="Massachusetts"').reset_index()
#ds_NY = df.query('state=="New York"').reset_index()
#ds_IL = df.query('state=="Illinois"').reset_index()
#ds_CA
#Map on county level # does not work :( api does not respone, url2 dataset to large ...
#import branca
#import json
#import requests
#url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
#url2 = 'https://public.opendatasoft.com/explore/embed/dataset/us-county-boundaries/table'
#county_data = f'{url}/us_county_data.csv'
#county_geo = f'{url2}/us_counties_20m_topo.json'

#colorscale = branca.colormap.linear.YlOrRd_09.scale(0, 50e3)
#ds_CA = df.query('state=="California"').groupby('fips')['cases'].max()
#cs_CA_o = df.set_index('fips')
#def style_function(feature):     #import of polygons for counties does not work somehow...
#    ob = ds_CA_o.get(int(feature['id'][-5:]), None)
#    return {
#        'fillOpacity': 0.5,
#        'weight': 0,
#        'fillColor': '#white' if ob is None else colorscale(ob)
#    }
#
#
#m = folium.Map(
#    location=[34, -118],
#    tiles='cartodbpositron',
#    zoom_start=7
#)

#folium.TopoJson(
#   json.loads(requests.get(county_geo).text),
#   'objects.us_counties_20m',
#   style_function=style_function
#).add_to(m)
#
#m
#Model for CA
#m = Prophet(interval_width=0.95)
#m.fit(cases_CA)
#future = m.make_future_dataframe(periods=30) #prediction period 1 month = 30 days
#future.tail()

#forecast_CA = m.predict(future)
#forecast_CA[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#cases_forecast_plot_CA = m.plot(forecast)
#forecast = m.predict(future)#
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#cases_forecast_plot = m.plot(forecast)


#map = folium.Map(location=(20,70), zoom_start=4, titles='Stamenterrain')

#for lat, lon, value, name in zip(df['lat'], df['long'], df['cases'], df['county'] folium.CircleMarker([lat,lon], radius=value, popup=['county']))

#map


#cases_ca_agg = cases_ca.groupby('date')['cases'].sum().sort_values(ascending=False).to_frame()#
#cases_ca_agg.style.background_gradient(cmap='Reds')
#cases_ca = cases_california[["date","cases"]]#
#cases_ca.head(10)

#dataframe = pd.read_csv("/kaggle/input/covid19-us-county-jhu-data-demographics/covid_us_county.csv")
#
#df = dataframe

#df.dropna() # deletes missing values

#df2 = df

#df2.pivot_table(values="Loan_Status", index=["Credit_History"], aggfunc=lambda x: x.map({"Y": 1, "N":0}).mean()).plot(kind="bar")

#df2.head(-10)
#cases_CA = df_CA[['date','cases']]#
#deaths_CA = df_CA[['date','deaths']]
#cases_CA.head(10)

#highest = df.sort_values('cases').drop_duplicates('state',keep='last').tail(5)
#grouped = highest.groupby("state")["cases"].plot(legend=True)
#grouped = df.groupby("state")["cases"].plot(legend=True)
#highest.plot(x="date", y="cases")

#plt.show()
#total deaths 
#import pandas as pd
#import matplotlib.pyplot as plt
#dataframe = pd.read_csv("/kaggle/input/covid19-us-county-jhu-data-demographics/covid_us_county.csv")
#df = dataframe

#highest = df.sort_values('deaths').drop_duplicates('state',keep='last').tail(5)
#highest.plot(kind='bar', x='state', y='deaths')
#plt.show()

#most_cases = df.sort_values('cases').drop_duplicates('state',keep='last').tail(5)
#most_cases.plot(kind='bar', x='state', y='cases')
#plt.show()



#df_new = pd.DataFrame(
#{"state" : [df['state']],
#"date" : [df['date']],
#"cases": [df['cases']]},
#index = [df['fips']])

#df_new.head(10)
#df_new2 = pd.DataFrame(
#{"date" : [df['date']],
#"cases": [df['cases']]},
#index = [df['state']])

#df_new2.head(10)
#df.describe
# prophet works only with 2 colums
#cases_CA.colums = ['ds','y']
#cases_CA.rename(columns = {'date':'ds'})
#cases_CA.rename(columns = {'cases':'y'})
#cases_CA['ds'] = pd.to_datetime(cases_CA['ds'])
#cases_CA.tail()

#import matplotlib.pyplot as plt
#import datetime as dtt
#x = df['date']
#y = df['cases']
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(x, y, color='lightblue')
#ax.legend(loc='best')
#plt.show

#dataframe.boxplot(column="cases", by="state", figsize=(15,8))#
#cases_CA.rename(columns={"A": "a", "B": "c"})
#cases_CA.colums = ['ds','y']
#cases = df.groupby('date').sum()['cases'].reset.index()
#deaths = df.groupby('date').sum()['deaths'].reset.index()
#fig = plt.figure(figsize=(8,4))
#axl = fig.add_subplot(121)
#axl.set_xlabel("date")
#axl.set_ylabel("cases")
#axl.set_title("Total cases")
#dataframe1.value_counts().plot(kind="bar")
# Attempt Join Dataframes
#demo = pd.read_csv("/kaggle/input/covid19-us-county-jhu-data-demographics/us_county.csv")
#demo.head(10)

#demo['median_age_state'] = 

#demo.pivot(index ='fips', columns ='state', values ='median_age') 
    