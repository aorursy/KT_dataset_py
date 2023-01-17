! pip install pycountry-convert
# ! pip install calmap
# ! pip install -Uq watermark
import torch

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib import ticker 
import pycountry_convert as pc
import folium
import branca
from datetime import datetime, timedelta,date
from scipy.interpolate import make_interp_spline, BSpline
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams

from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

import plotly.express as px
import plotly.offline as py
import plotly.graph_objs as gos
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
import json, requests
# import calmap
import operator 
import folium

from keras.layers import Input, Dense, Activation, LeakyReLU
from keras import models
from fbprophet import Prophet
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from keras.optimizers import RMSprop, Adam

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
lastupdate_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/07-07-2020.csv')
latest_data = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_time.csv",parse_dates=['Last_Update'])
Country_df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv")
# Read Top 2 Rows of latest_data 
latest_data.head(2)
# Read Top 2 Rows of deaths_df 
deaths_df.head(2)
# Read Top 2 Rows of deaths_df 
recoveries_df.head(2)
# Read Top 2 Rows of lastupdate_data
lastupdate_data.head(2)
# Read Top 2 Rows of confirmed_df 
confirmed_df.head(2)
# Read Top 2 Rows of Country_df 
Country_df.head(2)
confirmed_df = confirmed_df.rename(columns={"Province/State":"state","Country/Region": "country"})
deaths_df = deaths_df.rename(columns={"Province/State":"state","Country/Region": "country"})
recoveries_df = recoveries_df.rename(columns={"Province/State":"state","Country/Region": "country"})
Country_df = Country_df.rename(columns={"Country_Region": "country"})
# Changing the conuntry names as required by pycountry_convert Lib
confirmed_df.loc[confirmed_df['country'] == "US", "country"] = "USA"
deaths_df.loc[deaths_df['country'] == "US", "country"] = "USA"
Country_df.loc[Country_df['country'] == "US", "country"] = "USA"
recoveries_df.loc[recoveries_df['country'] == "US", "country"] = "USA"
latest_data.loc[latest_data['Country_Region'] == "US", "Country_Region"] = "USA"
dates1 = confirmed_df.columns[4:]

confirmed_df1 = confirmed_df.melt(id_vars=['state', 'country', 'Lat', 'Long'], 
                            value_vars=dates1, var_name='Date', value_name='Confirmed')

deaths_df1 = deaths_df.melt(id_vars=['state', 'country', 'Lat', 'Long'], 
                            value_vars=dates1, var_name='Date', value_name='Deaths')

recoveries_df1 = recoveries_df.melt(id_vars=['state', 'country', 'Lat', 'Long'], 
                            value_vars=dates1, var_name='Date', value_name='Recovered')
# getting all countries
countries = np.asarray(confirmed_df["country"])
countries1 = np.asarray(Country_df["country"])

# Continent_code to Continent_names
continents = {
    'NA': 'North America',
    'SA': 'South America', 
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
    'EU' : 'Europe',
    'na' : 'Others'
}

# Defininng Function for getting continent code for country.
def country_to_continent_code(country):
    try:
        return pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
    except :
        return 'na'

#Collecting Continent Information
confirmed_df.insert(2,"continent", [continents[country_to_continent_code(country)] for country in countries[:]])
deaths_df.insert(2,"continent",  [continents[country_to_continent_code(country)] for country in countries[:]])
Country_df.insert(1,"continent",  [continents[country_to_continent_code(country)] for country in countries1[:]])
latest_data.insert(1,"continent",  [continents[country_to_continent_code(country)] for country in latest_data["Country_Region"].values])
#recoveries_df.insert(2,"continent",  [continents[country_to_continent_code(country)] for country in countries[:]] )  
# Handaling Missing data
confirmed_df = confirmed_df.replace(np.nan, '', regex=True)
deaths_df = deaths_df.replace(np.nan, '', regex=True)
Full_data = pd.merge(left=confirmed_df1, right=deaths_df1, how='left',
                      on=['state', 'country', 'Date', 'Lat', 'Long'])
Full_data = pd.merge(left=Full_data, right=recoveries_df1, how='left',
                      on=['state', 'country', 'Date', 'Lat', 'Long'])
# Active Case = confirmed - deaths - recovered
Full_data['Active'] = Full_data['Confirmed'] - Full_data['Deaths'] - Full_data['Recovered']
Full_data.head(5)
# Handaling Missing data
Full_data=Full_data.dropna(subset=['Long'])
Full_data=Full_data.dropna(subset=['Lat'])
# latest condensed
#latest_grouped = lastupdate_data.groupby('Country_Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
latest_grouped = Country_df.groupby('country')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 
china_cases = [] 
italy_cases = []
usa_cases = [] 
spain_cases = [] 

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    
    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum-death_sum-recovered_sum)
    
    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)

    # case studies 
    china_cases.append(confirmed_df[confirmed_df['country']=='China'][i].sum())
    italy_cases.append(confirmed_df[confirmed_df['country']=='Italy'][i].sum())
    usa_cases.append(confirmed_df[confirmed_df['country']=='USA'][i].sum())
    spain_cases.append(confirmed_df[confirmed_df['country']=='Spain'][i].sum())
def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 

world_daily_increase = daily_increase(world_cases)
china_daily_increase = daily_increase(china_cases)
italy_daily_increase = daily_increase(italy_cases)
usa_daily_increase = daily_increase(usa_cases)
spain_daily_increase = daily_increase(spain_cases)
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)
# Cases over time
def scatterPlotCasesOverTime(df, country):
    plot = make_subplots(rows=2, cols=2, subplot_titles=("Comfirmed", "Deaths", "Recovered", "Active"))

    subPlot1 = gos.Scatter(
                    x=df['Date'],
                    y=df['Confirmed'],
                    name="Confirmed",
                    line_color='orange',
                    opacity=0.8)

    subPlot2 = gos.Scatter(
                    x=df['Date'],
                    y=df['Deaths'],
                    name="Deaths",
                    line_color='red',
                    opacity=0.8)

    subPlot3 = gos.Scatter(
                    x=df['Date'],
                    y=df['Recovered'],
                    name="Recovered",
                    line_color='green',
                    opacity=0.8)
    
    subPlot4 = gos.Scatter(
                    x=df['Date'],
                    y=df['Active'],
                    name="Active",
                    line_color='blue',
                    opacity=0.8)

    plot.append_trace(subPlot1, 1, 1)
    plot.append_trace(subPlot2, 1, 2)
    plot.append_trace(subPlot3, 2, 1)
    plot.append_trace(subPlot4, 2, 2)
    plot.update_layout(template="ggplot2", title_text = country + '<b> - Spread of the nCov Over Time</b>')

    plot.show()
# For Future forcasting
days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]
df_countries_cases = Country_df.copy().drop(['Lat','Long_','continent','Last_Update'],axis =1)
df_countries_cases.index = df_countries_cases["country"]
df_countries_cases = df_countries_cases.drop(['country'],axis=1)

df_continents_cases = Country_df.copy().drop(['Lat','Long_','country','Last_Update'],axis =1)
df_continents_cases = df_continents_cases.groupby(["continent"]).sum()
df_t = pd.DataFrame(pd.to_numeric(df_countries_cases.sum()),dtype=np.float64).transpose()
df_t["Mortality Rate (per 100)"] = np.round(100*df_t["Deaths"]/df_t["Confirmed"],2)
df_t.style.background_gradient(cmap='Wistia',axis=1).format("{:.0f}",subset=["Confirmed"])
# For each single countries
unique_countries =  list(Country_df['country'].unique())
country_confirmed_cases = []
country_death_cases = [] 
country_recovery_cases = []
country_active_cases = []
country_mortality_rate = [] 


no_cases = []
for i in unique_countries:
    cases = Country_df[Country_df['country']==i]['Confirmed'].sum()
    if cases > 0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
for i in no_cases:
    unique_countries.remove(i)
# sort countries by the number of confirmed cases
unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = Country_df[Country_df['country']==unique_countries[i]]['Confirmed'].sum()
    country_death_cases.append(Country_df[Country_df['country']==unique_countries[i]]['Deaths'].sum())
    country_recovery_cases.append(Country_df[Country_df['country']==unique_countries[i]]['Recovered'].sum())
    country_active_cases.append(Country_df[Country_df['country']==unique_countries[i]]['Active'].sum())
    country_mortality_rate.append((country_death_cases[i]/country_confirmed_cases[i])*100)
country_df = pd.DataFrame({'Country Name': unique_countries, 'Number of Confirmed Cases': country_confirmed_cases,
                          'Number of Deaths': country_death_cases, 'Number of Recoveries' : country_recovery_cases,
                           'Number of Active': country_active_cases, 'Mortality Rate': country_mortality_rate})

# number of cases per country/region
country_df.style.background_gradient(cmap="Wistia", subset=['Number of Confirmed Cases'])\
                .background_gradient(cmap="Reds", subset=['Number of Deaths'])\
                .background_gradient(cmap="summer", subset=['Number of Recoveries'])\
                .background_gradient(cmap="OrRd", subset=['Number of Active'])
adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, world_cases)
plt.title('No. of Coronavirus Cases Over Time', size=25)
plt.xlabel('Days Since 1/22/2020', size=25)
plt.ylabel('No. of Cases', size=25)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, world_daily_increase)
plt.title('World Daily Increases in Confirmed Cases', size=25)
plt.xlabel('Days Since 1/22/2020', size=25)
plt.ylabel('No. of Cases', size=25)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, world_cases, color='b')
plt.plot(adjusted_dates, total_deaths, color='r')
plt.plot(adjusted_dates, total_recovered, color='green')
plt.title('No. of Coronavirus Total cases vs Death cases vs Recovered Cases', size=25)
plt.legend(['Cases','Death','Recoveries' ], loc='best', fontsize=20)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
mean_mortality_rate = np.mean(mortality_rate)
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, mortality_rate, color='orange')
plt.axhline(y = mean_mortality_rate,linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time', size=30)
plt.legend(['mortality rate', 'y='+str(mean_mortality_rate)], prop={'size': 20})
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Mortality Rate', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
df_countries_cases.groupby('country')['Confirmed'].sum().sort_values(ascending=False)[:10]
f = plt.figure(figsize=(10,5))
f.add_subplot(111)
out = ""
plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Confirmed')["Confirmed"].index[-10:],df_countries_cases.sort_values('Confirmed')["Confirmed"].values[-10:],color="deepskyblue")
#color="darkcyan"
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top 10 Countries Under Corona Confirmed Cases",fontsize=20)
plt.grid(alpha=0.3)
plt.savefig(out+'Top 10 Countries (Confirmed Cases).png')
df_countries_cases.groupby('country')['Deaths'].sum().sort_values(ascending=False)[:10]
f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Deaths')["Deaths"].index[-10:],df_countries_cases.sort_values('Deaths')["Deaths"].values[-10:],color="crimson")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Deaths Cases",fontsize=18)
plt.title("Top 10 Countries Under Corona Deaths Cases",fontsize=20)
plt.grid(alpha=0.3,which='both')
plt.savefig(out+'Top 10 Countries (Deaths Cases).png')
df_countries_cases.groupby('country')['Recovered'].sum().sort_values(ascending=False)[:10]
f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_countries_cases.sort_values('Recovered')["Recovered"].index[-10:],df_countries_cases.sort_values('Recovered')["Recovered"].values[-10:],color="yellowgreen")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Recovered Cases",fontsize=18)
plt.title("Top 10 Countries Under Corona Recovered Cases",fontsize=20)
plt.grid(alpha=0.3,which='both')
plt.savefig(out+'Top 10 Countries (Recovered Cases).png')
plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, usa_daily_increase)
plt.title('US Daily Increases in Confirmed Cases', size=25)
plt.xlabel('Days Since 1/22/2020', size=25)
plt.ylabel('No. of Cases', size=25)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, china_daily_increase)
plt.title('China Daily Increases in Confirmed Cases', size=25)
plt.xlabel('Days Since 1/22/2020', size=25)
plt.ylabel('No. of Cases', size=25)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, italy_daily_increase)
plt.title('Italy Daily Increases in Confirmed Cases', size=25)
plt.xlabel('Days Since 1/22/2020', size=25)
plt.ylabel('No. of Cases', size=25)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, spain_daily_increase)
plt.title('Spain Daily Increases in Confirmed Cases', size=25)
plt.xlabel('Days Since 1/22/2020', size=25)
plt.ylabel('No. of Cases', size=25)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, usa_cases)
plt.plot(adjusted_dates, china_cases)
plt.plot(adjusted_dates, italy_cases)
plt.plot(adjusted_dates, spain_cases)
plt.title('No. of Coronavirus Cases', size=25)
plt.xlabel('Days Since 1/22/2020', size=22)
plt.ylabel('No. of Cases', size=25)
plt.legend(['US', 'China', 'Italy', 'Spain'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Confirmed
fig = px.choropleth(latest_grouped, locations="country", 
                    locationmode='country names', color=np.log(latest_grouped["Confirmed"]), 
                    hover_name="country", hover_data=['Confirmed'],
                    color_continuous_scale="Sunsetdark", 
                    title='Countries with Confirmed Cases')
fig.update(layout_coloraxis_showscale=False)
fig.show()
# Deaths
temp = latest_grouped[latest_grouped['Deaths']>0]
fig = px.choropleth(temp, 
                    locations="country", locationmode='country names',
                    color=np.log(temp["Deaths"]), hover_name="country", 
                    color_continuous_scale="Peach", hover_data=['Deaths'],
                    title='Countries with Deaths Reported')
fig.update(layout_coloraxis_showscale=False)
fig.show()
formated_gdf = Full_data.groupby(['Date', 'country'])['Confirmed', 'Deaths'].max().reset_index()
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.2)

fig = px.scatter_geo(formated_gdf, locations="country", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="country", 
                     range_color= [0, max(formated_gdf['Confirmed'])+2], animation_frame="Date", 
                     title='Spread over time')
fig.update(layout_coloraxis_showscale=False)
fig.show()
'''
# World wide

m = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=1, max_zoom=4, zoom_start=1)

for i in range(0, len(Full_data)):
    folium.Circle(
        location=[Full_data.iloc[i]['Lat'], Full_data.iloc[i]['Long']],
        color='crimson', 
        tooltip =   '<li><bold>Country : '+str(Full_data.iloc[i]['country'])+
                    '<li><bold>Province : '+str(Full_data.iloc[i]['state'])+
                    '<li><bold>Confirmed : '+str(Full_data.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(Full_data.iloc[i]['Deaths']),
        radius=int(Full_data.iloc[i]['Confirmed'])**1.1).add_to(m)
m
'''
df_usa = lastupdate_data.loc[lastupdate_data["Country_Region"]== "US"]
df_usa.head(2)
#df_usa = df_usa.rename(columns={"Admin2":"County"})
total = df_usa.sum()
total.name = "Total"
pd.DataFrame(total).transpose().loc[:,["Confirmed","Deaths"]].style.background_gradient(cmap='Purples',axis=1)
df_usa.loc[:,["Confirmed","Deaths","Province_State"]].groupby(["Province_State"]).sum().sort_values("Confirmed",ascending=False).style.background_gradient(cmap='Blues',subset=["Confirmed"]).background_gradient(cmap='Reds',subset=["Deaths"])
f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_usa.groupby(["Province_State"]).sum().sort_values('Confirmed')["Confirmed"].index[-10:],df_usa.groupby(["Province_State"]).sum().sort_values('Confirmed')["Confirmed"].values[-10:],color="salmon")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Confirmed Cases",fontsize=18)
plt.title("Top 10 States: USA (Confirmed Cases)",fontsize=20)
plt.grid(alpha=0.3)
plt.savefig(out+'Top 10 States_USA (Confirmed Cases).png')
f = plt.figure(figsize=(10,5))
f.add_subplot(111)

plt.axes(axisbelow=True)
plt.barh(df_usa.groupby(["Province_State"]).sum().sort_values('Deaths')["Deaths"].index[-10:],df_usa.groupby(["Province_State"]).sum().sort_values('Deaths')["Deaths"].values[-10:],color="crimson")
plt.tick_params(size=5,labelsize = 13)
plt.xlabel("Deaths",fontsize=18)
plt.title("Top 10 States: USA (Deaths Cases)",fontsize=20)
plt.grid(alpha=0.3)
plt.savefig(out+'Top 10 States_USA (Deaths Cases).png')
Full_data.head()
df_usa_data = Full_data.loc[Full_data["country"]== "USA"]
#df_usa_data['Last_Update'] =  pd.to_datetime(df_usa_data['Last_Update'])
#df_usa_data['Last_Update'] = df_usa_data['Last_Update'].dt.date
#df_usa_data = df_usa_data.rename(columns={"Last_Update":"Date"})
df1 =  df_usa_data[['Date','Confirmed','Deaths','Recovered','Active']]
df1.head()

# USA - Cases over time
scatterPlotCasesOverTime(df1, "<b>USA</b>")
df_usa = df_usa.rename(columns={"Admin2":"County"})
df_usa = df_usa.replace(np.nan, 0, regex=True)
usa = folium.Map(location=[37, -102], tiles='cartodbpositron', min_zoom=4, max_zoom=8, zoom_start=4)
for i in np.int32(np.asarray(df_usa[df_usa['Confirmed'] > 0].index)):
    folium.Circle(
        location=[df_usa.loc[i]['Lat'], df_usa.loc[i]['Long_']],
        tooltip = "<h5 style='text-align:center;font-weight: bold'>"+df_usa.loc[i]['Province_State']+"</h5>"+
                    "<div style='text-align:center;'>"+str(np.nan_to_num(df_usa.loc[i]['County']))+"</div>"+
                    "<hr style='margin:10px;'>"+
                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
        "<li>Confirmed: "+str(df_usa.loc[i]['Confirmed'])+
        "<li>Active:   "+str(df_usa.loc[i]['Active'])+
        "<li>Recovered:   "+str(df_usa.loc[i]['Recovered'])+     
        "<li>Deaths:   "+str(df_usa.loc[i]['Deaths'])+
        "<li>Mortality Rate:   "+str(np.round(df_usa.loc[i]['Deaths']/(df_usa.loc[i]['Confirmed']+1)*100,2))

        ,
        radius=int((np.log2(df_usa.loc[i]['Confirmed']+1))*6000),
        color='yellowgreen',
        fill_color='red',
        fill=True).add_to(usa)

usa

state_geo = requests.get('https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json').json()
county_geo = requests.get('https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us_counties_20m_topo.json').json()
# county_geo
# binsurl = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
# county_data = f'{url}/us_county_data.csv'
# county_geo = f'{url}/us_counties_20m_topo.json'

data_temp = df_usa.groupby(["FIPS"]).sum().reset_index().drop(["Lat","Long_"],axis=1)
data_temp["Confirmed_log"] = np.log10(data_temp["Confirmed"]+1)

df_usa_series = data_temp.set_index('FIPS')['Confirmed_log']
colorscale = branca.colormap.linear.Reds_09.scale(0,data_temp["Confirmed_log"].max()-1)
# print(df_usa_series.max())
def style_function(feature):
    employed = df_usa_series.get(int(feature['id'][-5:]), 0)
    return {
        'fillOpacity': 0.5,
        'weight': 0,
        'fillColor': '#black' if employed is None else colorscale(employed)
    }


m = folium.Map(
    location=[37, -102],
    tiles='cartodbpositron',
    zoom_start=4,
    min_zoom=3,
    max_zoom=7
)

folium.TopoJson(
    county_geo,
    'objects.us_counties_20m',
    style_function=style_function
).add_to(m)
m
df_usa1 = confirmed_df.loc[confirmed_df["country"]== "USA"]
df_usa1.head(2)
temp = df_usa1.melt(value_vars=dates1, var_name='Date', value_name='Confirmed')
temp = temp.groupby('Date')['Confirmed'].sum().reset_index()
#Full_data[['Date','Confirmed']]
pr_data = pd.DataFrame(temp)

pr_data.columns = ['ds','y']
pr_data
m=Prophet()
m.fit(temp)
future=m.make_future_dataframe(periods=10)
forecast=m.predict(future)
forecast
cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm=cnfrm.tail(15)
cnfrm.columns = ['Date','Confirm']
cnfrm.head()
fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='Date',ylabel='Confirmed Count')
figure=m.plot_components(forecast)
df_usa1.head()
df_usa1 = df_usa1.iloc[:, 5:]
df_usa1.head()
daily_cases = df_usa1.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
daily_cases.head()
plt.plot(daily_cases)
plt.title("Cumulative daily cases");
daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
daily_cases.head()

plt.plot(daily_cases)
plt.title("Daily cases");
daily_cases.shape
train_data=daily_cases.iloc[:int(len(daily_cases)*0.8)] 
test_data=daily_cases.iloc[int(len(daily_cases)*0.8):]

train_data.shape
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))

test_data = scaler.transform(np.expand_dims(test_data, axis=1))
def train_test_split(daily_cases):
    size=int(len(daily_cases)*0.8)
    # for train data will be collected from each country's data which index is from 0-size (80%)
    x_train =daily_cases.drop(columns=['TargetValue']).iloc[0:size] 
    # for test data will be collected from each country's  data which index is from size to the end (20%)
    x_test = daily_cases.drop(columns=['TargetValue']).iloc[size:]
    y_train=daily_cases['TargetValue'].iloc[0:size] 
    y_test=daily_cases['TargetValue'].iloc[size:] 
    return x_train, x_test,y_train,y_test
# unique countries
country=list(set(daily_cases.country))
# loop each station and collect train and test data 
X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
for i in range(0,len(country)):
    df=data[['country']==country[i]]
    x_train, x_test,y_train,y_test=train_test_split(df)
    X_train.append(x_train)
    X_test.append(x_test)
    Y_train.append(y_train)
    Y_test.append(y_test)
# concat each train data from each station 
X_train=pd.concat(X_train)
Y_train=pd.DataFrame(pd.concat(Y_train))
# concat each test data from each station 
X_test=pd.concat(X_test)
Y_test=pd.DataFrame(pd.concat(Y_test))
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)
seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
class CoronaVirusPredictor(nn.Module):

  def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
    super(CoronaVirusPredictor, self).__init__()

    self.n_hidden = n_hidden
    self.seq_len = seq_len
    self.n_layers = n_layers

    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=n_hidden,
      num_layers=n_layers,
      dropout=0.5
    )

    self.linear = nn.Linear(in_features=n_hidden, out_features=1)

  def reset_hidden_state(self):
    self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
    y_pred = self.linear(last_time_step)
    return y_pred
def train_model(
  model, 
  train_data, 
  train_labels, 
  test_data=None, 
  test_labels=None
):
  loss_fn = torch.nn.MSELoss(reduction='sum')

  optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
  num_epochs = 800   # if its time consuming then set it to 500 or 100
  train_hist = np.zeros(num_epochs)
  test_hist = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.reset_hidden_state()

    y_pred = model(X_train)

    loss = loss_fn(y_pred.float(), y_train)

    if test_data is not None:
      with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred.float(), y_test)
      test_hist[t] = test_loss.item()

      if t % 1000 == 0:  
        print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
    elif t % 1000 == 0:
      print(f'Epoch {t} train loss: {loss.item()}')

    train_hist[t] = loss.item()
    
    optimiser.zero_grad()

    loss.backward()

    optimiser.step()
  
  return model.eval(), train_hist, test_hist
model = CoronaVirusPredictor(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist, test_hist = train_model(
  model, 
  X_train, 
  y_train, 
  X_test, 
  y_test
)
plt.plot(train_hist, label="Training loss")
plt.plot(test_hist, label="Test loss")
plt.ylim((0, 60))
plt.legend();
with torch.no_grad():
  test_seq = X_test[:1]
  preds = []
  for _ in range(len(X_test)):
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
true_cases = scaler.inverse_transform(
    np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()
plt.plot(
  daily_cases.index[:len(train_data)], 
  scaler.inverse_transform(train_data).flatten(),
  label='Historical Daily Cases'
)

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)], 
  true_cases,
  label='Real Daily Cases'
)

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)], 
  predicted_cases, 
  label='Predicted Daily Cases'
)

plt.legend();

scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(daily_cases, axis=1))

all_data = scaler.transform(np.expand_dims(daily_cases, axis=1))

all_data.shape
X_all, y_all = create_sequences(all_data, seq_length)

X_all = torch.from_numpy(X_all).float()
y_all = torch.from_numpy(y_all).float()

model = CoronaVirusPredictor(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist, _ = train_model(model, X_all, y_all)
DAYS_TO_PREDICT = 7

with torch.no_grad():
  test_seq = X_all[:1]
  preds = []
  for _ in range(DAYS_TO_PREDICT):
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
    
predicted_cases = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()
daily_cases.index[-1]
predicted_index = pd.date_range(
  start=daily_cases.index[-1],
  periods=DAYS_TO_PREDICT + 1,
  closed='right'
)

predicted_cases = pd.Series(
  data=predicted_cases,
  index=predicted_index
)

plt.plot(predicted_cases, label='Predicted Daily Cases')
plt.legend();
plt.plot(daily_cases, label='Historical Daily Cases')
plt.plot(predicted_cases, label='Predicted Daily Cases')
plt.legend();