import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
import geopandas as gpd
import time
import pylab as pl
from IPython import display
data = pd.read_csv("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")
data
data["location"].unique()
data_noworld = data[data["location"] != "World"]
data_onlycountries = data_noworld[data_noworld["location"] != "International"]
data_onlycountries["location"].unique()
data_onlycountries = data_onlycountries[["location", "date", "total_cases"]]
data_onlycountries
data_countrydate = data_onlycountries.groupby(["date", "location"]).sum().reset_index()
data_countrydate
fig = px.choropleth(data_countrydate, 
                    locations="location", 
                    locationmode = "country names",
                    color="total_cases", 
                    hover_name="location", 
                    color_continuous_scale="OrRd",
                    animation_frame="date"
                   )

fig.update_layout(
    title_text = "Total cases of coronavirus",
    title_x = 0.5,
    geo=dict(
        showframe = False,
        showcoastlines = False,
    ))
    
fig.show()
confirmed = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
confirmed.drop(["Province/State", "Lat", "Long"], axis=1, inplace=True)
confirmed.set_index("Country/Region", inplace=True)
dates = list(confirmed.columns)[1:]
confirmedportugal = list(confirmed.loc["Portugal"])[1:]
fig, ax = plt.subplots(figsize=(16, 9))
ax.bar(dates, confirmedportugal, color="dimgrey", label="Confirmed")
plt.xticks(list(np.arange(0, len(dates), 15)))
plt.legend(prop={'size': 20}, loc=2)
plt.show()
deaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
deaths.drop(["Province/State", "Lat", "Long"], axis=1, inplace=True)
deaths.set_index("Country/Region", inplace=True)
deathsportugal = list(deaths.loc["Portugal"])[1:]
fig, ax = plt.subplots(figsize=(16, 9))
ax.bar(dates, deathsportugal, color="red", label="Deaths")
plt.xticks(list(np.arange(0, len(dates), 15)))
plt.legend(prop={'size': 20}, loc=2)
plt.show()
recovered = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
recovered.drop(["Province/State", "Lat", "Long"], axis=1, inplace=True)
recovered.set_index("Country/Region", inplace=True)
recoveredportugal = list(recovered.loc["Portugal"])[1:]
fig, ax = plt.subplots(figsize=(16, 9))
ax.bar(dates, recoveredportugal, color="green", label="Recovered")
plt.xticks(list(np.arange(0, len(dates), 15)))
plt.legend(prop={'size': 20}, loc=2)
plt.show()
active = []
for i in range(len(confirmedportugal)):
    active.append(confirmedportugal[i] - recoveredportugal[i] - deathsportugal[i])
fig, ax = plt.subplots(figsize=(16, 9))
ax.bar(dates, active, label="Active")
plt.xticks(list(np.arange(0, len(dates), 15)))
plt.legend(prop={'size': 20}, loc=2)
plt.show()
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(dates, deathsportugal, color="red", label="Deaths")
plt.fill_between(dates, deathsportugal, color="red")
ax.plot(dates, active, color="skyblue", label="Active")
plt.fill_between(dates, active, color="skyblue", alpha=0.6)
ax.plot(dates, recoveredportugal, color="mediumspringgreen", label="Recovered")
plt.fill_between(dates, recoveredportugal, color="mediumspringgreen", alpha=0.4)
ax.plot(dates, confirmedportugal, color="lightgrey", label="Confirmed")
plt.fill_between(dates, confirmedportugal, color="lightgrey", alpha=0.6)
plt.legend(prop={'size': 20})
plt.xticks(list(np.arange(0, len(dates), 15)))
plt.show()
ptdata = gpd.read_file("https://raw.githubusercontent.com/nmota/caop_GeoJSON/master/ContinenteConcelhos.geojson")
ptdata.head()
confirmados = pd.read_csv("https://raw.githubusercontent.com/dssg-pt/covid19pt-data/master/data_concelhos.csv")
confirmados
for num in range(len(confirmados)-1):
  confirmadosdia = confirmados.iloc[num]
  confirmadosdia = confirmadosdia[1:]
  confirmadosdia.fillna(0, inplace=True)
  confirmadosdia=pd.DataFrame(confirmadosdia)
  confirmadosdia.reset_index(inplace=True)
  confirmadosdia.rename({"index": "Concelho"}, axis=1, inplace=True)
  confirmadosdia.rename({num: "Casos confirmados"}, axis=1, inplace=True)
  confirmadosgeo = ptdata.merge(confirmadosdia, on="Concelho")
  confirmadosgeo = confirmadosgeo[["Concelho", "geometry", "Casos confirmados"]]
  confirmadosgeo.plot(column = "Casos confirmados",
                         cmap='YlGnBu',
                         figsize=(16,10),
                         legend=True,
                         edgecolor="black")
  display.display(pl.gcf())
  display.clear_output(wait=True)
  plt.close()
  time.sleep(0.1)