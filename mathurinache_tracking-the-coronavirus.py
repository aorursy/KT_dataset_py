import pandas as pd
df = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

df.Date = pd.to_datetime(df.Date)

df.head()
total = df.groupby(['Date']).sum().loc[:,['Confirmed','Deaths','Recovered']].reset_index()

total.head()
print('A total of: %s  cases were confirmed.' %(total.Confirmed.sum()))

print('A total of: %s (%s %%) deaths were recorded.' %(total.Deaths.sum(), ((100*total.Deaths.sum())/total.Confirmed.sum())))

print('A total of: %s (%s %%) recovered from the infection.' %(total.Recovered.sum(), ((100*total.Recovered.sum())/total.Confirmed.sum())))
from folium.plugins import HeatMap

import folium





center_lat = df[df['Province/State']=='Hubei'].iloc[0].Lat

center_lon = df[df['Province/State']=='Hubei'].iloc[0].Long





hmp = folium.Map(

    location=[center_lat, center_lon],

    zoom_start=2,

    tiles='OpenStreetMap', 

    width='100%')



HeatMap(data=df[['Lat', 'Long']].groupby(['Lat', 'Long']).count().reset_index().values.tolist(), radius=10, max_zoom=13).add_to(hmp)



hmp

from datetime import datetime, timedelta

from folium.plugins import HeatMapWithTime





df_day_list = []



for day in df.Date.sort_values().unique():

    

    temp = df.loc[df.Date == day, ['Lat', 'Long', 'Confirmed']].groupby(['Lat', 'Long']).sum().reset_index()    

    df_day_list.append(temp[temp.Confirmed>0].reset_index(drop=True).values.tolist())





time_index = [(df.Date[0] + k * timedelta(1)).strftime('%Y-%m-%d') for

    k in range(len(df_day_list))

]



def generateBaseMap(default_location=[center_lat, center_lon], default_zoom_start=2):

    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    return base_map





base_map = generateBaseMap(default_zoom_start=2)



HeatMapWithTime(df_day_list, 

                radius=10, 

                index=time_index,

                auto_play=True,

                use_local_extrema=False).add_to(base_map)

base_map
total['Currently_infected'] = total.Confirmed-(total.Deaths+total.Recovered)
import matplotlib.pyplot as plt

import seaborn as sns



sns.set_palette('viridis')



fig, axes = plt.subplots(1, 2, sharex=True, figsize=(14,5))





sns.lineplot(x='Date',y='Confirmed', data=total, label='Confirmed', ax=axes[0])

sns.lineplot(x='Date',y='Deaths', data=total, label='Deaths', ax=axes[0])

sns.lineplot(x='Date',y='Recovered', data=total, label='Recovered', ax=axes[0])

sns.lineplot(x='Date',y='Currently_infected', data=total, label='Current infections', ax=axes[0])







sns.lineplot(x='Date',y='Confirmed', data=total, label='Confirmed', ax=axes[1])

sns.lineplot(x='Date',y='Deaths', data=total, label='Deaths', ax=axes[1])

sns.lineplot(x='Date',y='Recovered', data=total, label='Recovered', ax=axes[1])

sns.lineplot(x='Date',y='Currently_infected', data=total, label='Current infections', ax=axes[1])



axes[1].set_yscale('log')



axes[0].title.set_text('Absolute numbers linear')

axes[1].title.set_text('Absolute numbers log')



axes[0].set( ylabel='Absolute numbers')

axes[1].set( ylabel='Log')



plt.sca(axes[0])

plt.xticks(rotation=40)



plt.sca(axes[1])

plt.xticks(rotation=40)



plt.legend()

sns.despine()

plt.tight_layout()



# plt.savefig('log_comparison.png')

country_sums = df.groupby('Country/Region').sum().loc[:,['Confirmed', 'Deaths', 'Recovered']]
country_sums['deathrate'] = (100*country_sums.Deaths)/country_sums.Confirmed

deaths = country_sums[country_sums.deathrate>0]
country_sums.sort_values('deathrate', ascending = False)
import matplotlib.pyplot as plt

from matplotlib import cm

from math import log10



data = deaths.sort_values('deathrate',ascending=False)





labels = list(data.index)

data = list(data.deathrate.values.ravel())

#number of data points

n = len(data)

#find max value for full ring

k = 10 ** int(log10(max(data)))

m = k * (1 + max(data) // k)



#radius of donut chart

r = 1.5

#calculate width of each ring

w = r / n 



#create colors along a chosen colormap

colors = [cm.Reds_r(i / n) for i in range(n)]



#create figure, axis

fig, ax = plt.subplots(figsize=(8,8))

ax.axis("equal")



#create rings of donut chart

for i in range(n):

    #hide labels in segments with textprops: alpha = 0 - transparent, alpha = 1 - visible

    innerring, _ = ax.pie([m - data[i], data[i]], radius = r - i * w, startangle = 90, labels = ["", labels[i]], labeldistance = 1 - 1 / (1.5 * (n - i)), textprops = {"alpha": 0}, colors = ["white", colors[i]])

    plt.setp(innerring, width = w, edgecolor = "white")



plt.legend( bbox_to_anchor= (1.2, 0.85))

plt.tight_layout()

plt.savefig('Death_rates.png')

plt.show()

from folium.plugins import MarkerCluster



mp = folium.Map(

    location=[center_lat, center_lon],

    zoom_start=2,

    tiles='OpenStreetMap', 

    width='100%')



mp.add_child(folium.LatLngPopup())



marker_cluster = MarkerCluster().add_to(mp)



for i, row in df.iterrows():

    name = row.Date

    lat = row.Lat

    lon = row.Long

    opened = row.Confirmed

    

    # HTML here in the pop up 

    popup = '<b>{}</b></br><i>setup date = {}</i>'.format(name, opened)

    

    folium.Marker([lat, lon], popup=popup, tooltip=name).add_to(marker_cluster)
mp