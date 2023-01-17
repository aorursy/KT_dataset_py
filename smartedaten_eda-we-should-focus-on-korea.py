from IPython.display import Image
Image("../input/coronavirus-images-data/coronavirus.jpg")
#import dependencies

from IPython.display import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
%matplotlib inline

from plotly.offline import iplot
from plotly import tools
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)

from bokeh.plotting import output_notebook, figure, show
from bokeh.models import ColumnDataSource, Div, Select, Button, ColorBar, CustomJS
from bokeh.layouts import row, column, layout
from bokeh.transform import cumsum, linear_cmap
from bokeh.palettes import Blues8, Spectral3
from bokeh.plotting import figure, output_file, show

import folium 
from folium import plugins
#plt.style.use("fivethirtyeight")# for pretty graphs

import os
#os.chdir("/data/shared/ch00tnb/Corona_datavis")

from datetime import datetime, timedelta

plt.style.use('dark_background')


#Data Sources
#Coronavirus: https://ourworldindata.org/coronavirus-source-data
#Geocode by country: https://developers.google.com/public-data/docs/canonical/countries_csv
#Population by country: https://data.worldbank.org/indicator/sp.pop.totl

#read in current data
full_data = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/full_data.csv")

#read in population data
population_data = pd.read_csv("../input/coronavirus-images-data/API_SP.POP.TOTL_DS2_en_csv_v2_887275.csv", skiprows=3)

#overwrite old with current data
#full_data.to_csv("../input/coronavirus-images-data/full_data.csv")
full_data.head()
full_data.tail()
print("External Data")
print(f"Earliest Entry: {full_data['date'].min()}")
print(f"Last Entry:     {full_data['date'].max()}")
print(f"Total Days:     {pd.to_datetime(full_data['date']).max() - pd.to_datetime(full_data['date']).min()}")
print("rows and columns")
full_data.shape
#data manipulations

# only use population information of latest year and rename columns
population_data_current = population_data.loc[:,["Country Name", "2018"]]
population_data_current = population_data_current.rename(columns={'Country Name': 'location', '2018': 'population_2018'})

#merge dataset with population information
full_data = full_data.merge(population_data_current, how="left", on="location")

#get current date and yesterday's date since data does not always represents today's date.
d = datetime.today()
current_date = datetime.strftime(datetime.now() - timedelta(0), '%Y-%m-%d')
yesterdays_date = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')

#get dates and days
full_data["days"] = pd.to_timedelta(pd.to_datetime(full_data["date"]) - pd.to_datetime("2019-12-31")).dt.days
full_data["date"] = full_data["date"].astype('O')

#create relative numbers for the countries
full_data["new_cases/1M_population"] = full_data["new_cases"] / full_data["population_2018"] * 1000000
full_data["new_deaths/1M_population"] = full_data["new_deaths"] / full_data["population_2018"] * 1000000
full_data["total_cases/1M_population"] = full_data["total_cases"] / full_data["population_2018"] * 1000000
full_data["total_deaths/1M_population"] = full_data["total_deaths"] / full_data["population_2018"] * 1000000

#define interesting countries
interesting_countries_list = ["United States", "China", "Italy", "Germany", "France", "Spain", "South Korea", "Switzerland"]

#create new dataframes based for plotting.
#get only current date
full_data_today = full_data[full_data["date"] == full_data["date"].max()]
full_data_today_sorted = full_data_today.sort_values("total_cases", ascending = False).set_index(['date'])
interesting_countries_data_today = full_data_today_sorted[full_data_today_sorted["location"].isin(interesting_countries_list)]
interesting_countries_data_today = interesting_countries_data_today.sort_values("new_cases", ascending = False)
interesting_countries_data_today_new_cases = interesting_countries_data_today.groupby("location")["new_cases"].sum()

#create new dataframes based for plotting. Referring to interesting countries and cases over 200
full_data_interesting_countries = full_data.loc[(full_data["location"].isin(interesting_countries_list)) & (full_data["total_cases"] > 200)]
full_data_interesting_countries_long = full_data.loc[(full_data["location"].isin(interesting_countries_list))]

#prepare comparison from ground0 for each country

country_dict = {}
for country in full_data_interesting_countries["location"].unique():
    value_length = full_data_interesting_countries[full_data_interesting_countries["location"] == country]["location"].count()
    country_dict.update({country : value_length})

ranges= []
for key in country_dict:
    for i in range(0,country_dict[key]):
        ranges.append(int(i))

full_data_interesting_countries["day0"] = ranges

minkey = min(country_dict, key=country_dict.get)
minvalue = country_dict[minkey]

full_data_interesting_countries2 = full_data_interesting_countries[full_data_interesting_countries["day0"] < minvalue]

#preparing data to plot by category, not by country
interesting_countries_df_date = full_data_interesting_countries_long.set_index("date")
interesting_countries_df = full_data_interesting_countries_long
Image("../input/coronavirus-images-data/korean_politicians.jpg")
fig = plt.figure(figsize=(15, 12))
ax = fig.gca()

plt.style.use(['dark_background'])

interesting_countries_data_today_total_cases = interesting_countries_data_today.groupby("location")["total_cases"].sum()
interesting_countries_data_today_total_cases.sort_values(ascending=True).plot(kind='barh', color=('Darkcyan'), alpha=0.8)

plt.title('Total cases by country', size=20, family = "Helvetica", weight= "bold")
plt.ylabel('Country Name', family = "Helvetica", size=15)
plt.xlabel(current_date, family = "Helvetica", size=13)
plt.xticks(rotation=45, family = "Helvetica", size=10)
plt.yticks(family = "Helvetica", size=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3.5)
    
#plt.grid(b=None)


plt.show()
fig = plt.figure(figsize=(14, 10))
ax = fig.gca()

interesting_countries_data_today_total_deaths = interesting_countries_data_today.groupby("location")["total_deaths"].sum()
interesting_countries_data_today_total_deaths.sort_values(ascending=True).plot(kind='barh', color=('Moccasin'), alpha=0.85)


plt.title('Total deaths by country', size=20, family = "Helvetica", weight= "bold")
plt.ylabel('Country Name', family = "Helvetica", size=15)
plt.xlabel(current_date, family = "Helvetica", size=13)
plt.xticks(rotation=45, family = "Helvetica", size=10)
plt.yticks(family = "Helvetica", size=14)


plt.style.use(['dark_background'])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3.5)
#plt.grid(b=None)

plt.show()
fig = plt.figure(figsize=(15, 12))
ax = fig.gca()

plt.style.use(['dark_background'])

interesting_countries_data_today_total_cases2 = interesting_countries_data_today.groupby("location")["total_cases/1M_population"].sum()
interesting_countries_data_today_total_cases2.sort_values(ascending=True).plot(kind='barh', color=('Cyan'), alpha=0.8)

plt.title('Total cases by country and per 1 million residents', size=20, family = "Helvetica", weight= "bold")
plt.ylabel('Country Name', family = "Helvetica", size=15)
plt.xlabel(current_date, family = "Helvetica", size=13)
plt.xticks(rotation=45, family = "Helvetica", size=10)
plt.yticks(family = "Helvetica", size=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3.5)
    
#plt.grid(b=None)


plt.show()
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
#plt.plot(full_data_interesting_countries.groupby(["days"]).sum()["total_cases"] )
plt.plot(full_data_interesting_countries.groupby(['days','location']).sum()['total_cases'].unstack())
#plt.plot(interesting_countries_df.groupby(['date','location']).sum()['total_cases'].unstack())
#full_data.loc[full_data["location"] == country].plot()

plt.title("total_cases by country", size=32)
plt.legend(labels =full_data_interesting_countries["location"].unique(), ncol=2, fontsize="large")
plt.xlabel("days since 2019/12/31" ,size=24)
plt.ylabel("# of cases", size=24)
#plt.grid(b=None)
#plt.gca(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
#plt.plot(full_data_interesting_countries.groupby(["days"]).sum()["total_cases"] )
plt.plot(full_data_interesting_countries.groupby(['days','location']).sum()['new_cases'].unstack())
#plt.plot(interesting_countries_df.groupby(['date','location']).sum()['total_cases'].unstack())
#full_data.loc[full_data["location"] == country].plot()

plt.title("new_cases by country", size=32)
plt.legend(labels =full_data_interesting_countries["location"].unique(), ncol=2, fontsize="large")
plt.xlabel("days since 2019/12/31" ,size=24)
plt.ylabel("# of cases", size=24)
#plt.grid(b=None)
#plt.gca(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
#plt.plot(full_data_interesting_countries.groupby(["days"]).sum()["total_cases"] )
plt.plot(full_data_interesting_countries.groupby(['days','location']).sum()['total_deaths'].unstack())
#plt.plot(interesting_countries_df.groupby(['date','location']).sum()['total_cases'].unstack())
#full_data.loc[full_data["location"] == country].plot()

plt.title("total_deaths by country", size=32)
plt.legend(labels =full_data_interesting_countries["location"].unique(), ncol=2, fontsize="large")
plt.xlabel("days since 2019/12/31" ,size=24)
plt.ylabel("# of cases", size=24)
#plt.grid(b=None)
#plt.gca(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
#plt.plot(full_data_interesting_countries.groupby(["days"]).sum()["total_cases"] )
plt.plot(full_data_interesting_countries.groupby(['days','location']).sum()['new_deaths'].unstack())
#plt.plot(interesting_countries_df.groupby(['date','location']).sum()['total_cases'].unstack())
#full_data.loc[full_data["location"] == country].plot()

plt.title("new_deaths by country", size=32)
plt.legend(labels =full_data_interesting_countries["location"].unique(), ncol=2, fontsize="large")
plt.xlabel("days since 2019/12/31" ,size=24)
plt.ylabel("# of cases", size=24)
#plt.grid(b=None)
#plt.gca(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()
xint = range(0, minvalue)

fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
#plt.plot(full_data_interesting_countries.groupby(["days"]).sum()["total_cases"] )
plt.plot(full_data_interesting_countries2.groupby(['day0','location']).sum()['total_cases'].unstack())
#plt.plot(interesting_countries_df.groupby(['date','location']).sum()['total_cases'].unstack())
#full_data.loc[full_data["location"] == country].plot()

plt.title("total_cases by country", size=32)
plt.legend(labels =full_data_interesting_countries["location"].unique(), ncol=2, fontsize="x-large")
plt.xlabel("days since # of cases in country reached more than total of 200" ,size=24)
plt.ylabel("# of cases", size=24)
plt.xticks(xint)
#plt.grid(b=None)
#plt.gca(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()
xint = range(0, minvalue)

fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
#plt.plot(full_data_interesting_countries.groupby(["days"]).sum()["total_cases"] )
plt.plot(full_data_interesting_countries2.groupby(['day0','location']).sum()['total_cases/1M_population'].unstack())
#plt.plot(interesting_countries_df.groupby(['date','location']).sum()['total_cases'].unstack())
#full_data.loc[full_data["location"] == country].plot()

plt.title("total_cases by country per 1M population", size=32)
plt.legend(labels =full_data_interesting_countries["location"].unique(), ncol=2, fontsize="x-large")
plt.xlabel("days since # of cases in country reached more than total of 200" ,size=24)
plt.ylabel("# of cases", size=24)
plt.xticks(xint)
#plt.grid(b=None)
#plt.gca(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
#plt.plot(full_data_interesting_countries.groupby(["days"]).sum()["total_cases"] )
plt.plot(full_data_interesting_countries[full_data_interesting_countries["days"]> 55].groupby(['days','location']).sum()['total_cases/1M_population'].unstack())
#plt.plot(interesting_countries_df.groupby(['date','location']).sum()['total_cases'].unstack())
#full_data.loc[full_data["location"] == country].plot()

plt.title("total_cases by country per 1M_population", size=32)
plt.legend(labels =full_data_interesting_countries["location"].unique(), ncol=2, fontsize="large")
plt.xlabel("days since 2019/12/31" ,size=24)
plt.ylabel("# of cases per 1M_population", size=24)
#plt.grid(b=None)
#plt.gca(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
#plt.plot(full_data_interesting_countries.groupby(["days"]).sum()["total_cases"] )
plt.plot(full_data_interesting_countries[full_data_interesting_countries["days"]> 55].groupby(['days','location']).sum()['total_deaths/1M_population'].unstack())
#plt.plot(interesting_countries_df.groupby(['date','location']).sum()['total_cases'].unstack())
#full_data.loc[full_data["location"] == country].plot()

plt.title("total_deaths by country per 1M_population", size=32)
plt.legend(labels =full_data_interesting_countries["location"].unique(), ncol=2, fontsize="large")
plt.xlabel("days since 2019/12/31" ,size=24)
plt.ylabel("# of cases per 1M_population", size=24)
#plt.grid(b=None)
#plt.gca(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
#plt.plot(full_data_interesting_countries.groupby(["days"]).sum()["total_cases"] )
plt.plot(full_data_interesting_countries[full_data_interesting_countries["days"]> 55].groupby(['days','location']).sum()['new_deaths/1M_population'].unstack())
#plt.plot(interesting_countries_df.groupby(['date','location']).sum()['total_cases'].unstack())
#full_data.loc[full_data["location"] == country].plot()

plt.title("new_deaths by country per 1M_population", size=32)
plt.legend(labels =full_data_interesting_countries["location"].unique(), ncol=2, fontsize="large")
plt.xlabel("days since 2019/12/31" ,size=24)
plt.ylabel("# of cases per 1M_population", size=24)
#plt.grid(b=None)
#plt.gca(visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.show()
Image("../input/coronavirus-images-data/care beds.jpg")
Image("../input/coronavirus-images-data/Korean Drive Through.jpg")
Image("../input/coronavirus-images-data/dongdaegu yeok.jpg")
Image("../input/coronavirus-images-data/Corona App.jpg")
Image("../input/coronavirus-images-data/voluntary lunch.jpg")
temp = full_data.groupby(['date', 'location'])['total_cases'].sum().reset_index()
temp['date'] = pd.to_datetime(temp['date'])
temp['date'] = temp['date'].dt.strftime('%m/%d/%Y')
temp['size'] = temp['total_cases'].pow(0.3) * 3.5

fig = px.scatter_geo(temp, locations="location", locationmode='country names', 
                     color="total_cases", size='size', hover_name="location", 
                     range_color=[1,100],
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Cases Over Time', color_continuous_scale="greens")
fig.show()