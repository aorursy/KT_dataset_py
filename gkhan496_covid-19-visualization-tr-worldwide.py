import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.basemap import Basemap
import folium


df_tr = pd.read_csv("../input/covid19-in-turkey/covid_19_data_tr.csv")
df_confirmed_tr = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_confirmed_tr.csv")
df_deaths_tr = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_deaths_tr.csv")
df_recovered_tr = pd.read_csv("../input/covid19-in-turkey/time_series_covid_19_recovered_tr.csv")
df_test_numbers = pd.read_csv("../input/covid19-in-turkey/test_numbers.csv")
df_tr
df_confirmed_tr
df_deaths_tr
df_recovered_tr
df_test_numbers
dates = df_confirmed_tr.iloc[:, 4:].columns
confirmed_num = df_confirmed_tr.iloc[:, 4:].values[0]
recovered_num = df_recovered_tr.iloc[:, 4:].values[0]
deaths_num = df_deaths_tr.iloc[:, 4:].values[0]
df_test_numbers = df_test_numbers.iloc[:, 4:].values[0]



fig = figure(num=None, figsize=(24, 10), dpi=100, facecolor='w', edgecolor='white')

linewidth = 3.5
linewidth1 = 0.6

plt.plot(dates,confirmed_num,color='purple', alpha=1, linewidth = linewidth, label = "CONFIRMED")
plt.scatter(dates,confirmed_num,color='white', alpha=1, linewidth = linewidth1)

plt.plot(dates,deaths_num,color='red', alpha=1, linewidth = linewidth, label = "DEATHS")
plt.scatter(dates,deaths_num,color='white', alpha=1, linewidth = linewidth1)

plt.plot(dates,recovered_num,color='#dfff00', alpha=1, linewidth = linewidth, label = "RECOVERED")
plt.scatter(dates,recovered_num,color='white', alpha=1, linewidth = linewidth1)

plt.plot(dates,df_test_numbers,color='white', alpha=1, linewidth = linewidth, label = "TESTNUM")
plt.scatter(dates,df_test_numbers,color='white', alpha=1, linewidth = linewidth1)

ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0, 0, 0))

plt.xticks(rotation='vertical')
plt.yticks(np.arange(0,max(confirmed_num) + 700, 8000))
plt.legend(loc = 0)
plt.grid(color='white', linestyle="--", linewidth=0.3, dash_joinstyle = "bevel")
plt.title('nCOV-2019 in Turkey')
plt.show()
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])
df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")
df_recovered = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
df_deaths = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)
df_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)
df_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)

df_confirmed.rename(columns={'Province/State':'Province'}, inplace=True)
df_recovered.rename(columns={'Province/State':'Province'}, inplace=True)
df_deaths.rename(columns={'Province/State':'Province'}, inplace=True)

last_day = df_confirmed.columns[-1]
df_confirmed[["Country",last_day]].sort_values(by=last_day, ascending=False).head(25)
df_recovered[["Country",last_day]].sort_values(by=last_day, ascending=False).head(25)
df_deaths[["Country",last_day]].sort_values(by=last_day, ascending=False).head(25)
# Veriseti belli aralıklarla güncelleniyor. Son vakalar aşağıda görüldüğü gibidir. 
# Sadece son vakalara bakarak bile virüsün hemen her kıtaya yayıldığını anlayabiliyoruz.
df.tail(10)
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()
recovered = df.groupby('Date').sum()['Recovered'].reset_index()


fig = figure(num=None, figsize=(50,12), dpi=120, facecolor='w', edgecolor='k')
linewidth = 3

plt.plot(confirmed["Date"],confirmed["Confirmed"],color='gray', alpha=1, linewidth = linewidth)
plt.plot(deaths["Date"],deaths["Deaths"],color='red', alpha=1, linewidth = linewidth)
plt.plot(recovered["Date"],recovered["Recovered"],color='yellow', alpha=1, linewidth = linewidth)

ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0, 0, 0))
plt.xticks(rotation='vertical')
plt.legend(title='Parameter where:')
plt.title('nCOV-2019')
plt.yticks(np.arange(0,max(confirmed["Confirmed"]) +700, 350000))
plt.grid(color='white', linestyle="--", linewidth=0.2, dash_joinstyle = "bevel")
plt.show()
world = df_confirmed[["Country",last_day]].sort_values(by='Country', ascending=True)
world['Country'].unique()
sorted_data = df_confirmed[["Country", "Lat", "Long", last_day]].sort_values(by=last_day, ascending=False)
long = sorted_data["Lat"]
lat = sorted_data["Long"]
countries = sorted_data["Country"]
confirmed = sorted_data[last_day]


# Make a data frame with the GPS of a few cities:
data = pd.DataFrame({
'lat': lat ,
'lon': long,
'name': countries
})

dpi = 96
plt.figure(figsize = (2600 / dpi, 1800 / dpi), dpi = dpi)

# A basic map
m = Basemap()
m.drawmapboundary(fill_color = '#A6CAE0', linewidth=0)
m.fillcontinents(color = 'red', alpha=0.3)
m.drawcoastlines(linewidth = 0.1, color = "black")

 
# Add a marker per city of the data frame!
m.plot(data['lat'], data['lon'], linestyle='none', marker="o", markersize=16, alpha=0.6, c="orange", markeredgecolor="black", markeredgewidth=1)

m = folium.Map(location=[4, -4], zoom_start=3,tiles='CartoDB positron')

for lat, lon, value, name in zip(sorted_data['Lat'], sorted_data['Long'], sorted_data[last_day], sorted_data['Country']):
                        folium.Circle([lat, lon],              
                        popup = ('<b color="">Country</b>: ' + str(name).capitalize() + '<br>'
                                 '<b>Confirmed</b>: '+ str(value) + '<br>'),
                        radius= 25,
                        color = '#D40C0C'
                       ).add_to(m)
m