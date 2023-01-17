import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series

import matplotlib.pyplot as plt

%matplotlib inline

from mpl_toolkits.basemap import Basemap

import warnings

warnings.filterwarnings('ignore')



volcanoes = pd.read_csv("../input/volcanic-eruptions/database.csv")

earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
#volcanoes.columns
#volcanoes.head()
#earthquakes.columns
#earthquakes.head()
def fig_p(data):

    series=Series(data).value_counts().sort_index()

    series.plot(kind='bar')
#the earthquakes dataset has nuclear explosions data in it so here i use only the earthquakes information

earthquakes_eq=pd.DataFrame()

earthquakes_eq=earthquakes[earthquakes['Type']=='Earthquake']
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

fig = plt.figure(figsize=(12,10))



longitudes_vol = volcanoes["Longitude"].tolist()

latitudes_vol = volcanoes["Latitude"].tolist()



longitudes_eq = earthquakes_eq["Longitude"].tolist()

latitudes_eq = earthquakes_eq["Latitude"].tolist()



x,y = m(longitudes_vol,latitudes_vol)

a,b= m(longitudes_eq,latitudes_eq)



plt.title("Volcanos areas (red) Earthquakes (green)")

m.plot(x, y, "o", markersize = 5, color = 'red')

m.plot(a, b, "o", markersize = 3, color = 'green')



m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='coral',lake_color='aqua')

m.drawmapboundary()

m.drawcountries()

plt.show()
#division of long&lat

def division(data):

    north_n=sum(data["Latitude"] >=30)

    middle_n=sum(np.logical_and(data["Latitude"]<30, data["Latitude"]>-30))

    south_n=sum(data["Latitude"]<= -30)

    #precentage

    total=north_n+middle_n+south_n

    north_p=north_n/total*100

    middle_p=middle_n/total*100

    south_p=south_n/total*100

    return north_n,middle_n,south_n,north_p,middle_p,south_p



volc=division(volcanoes)

eq=division(earthquakes_eq)



print("There are",volc[0],"volcanoes in latitude over 30N",volc[1],

      "in latitude between 30N and 30S and",volc[2],

      "in latitude over 30S. In precentage it is %.2f%%"% volc[3],",",

      "%.2f%%"% volc[4],"and","%.2f%%"% volc[5],"respectively.\n")



print("There were",eq[0],"earthquakes in latitude over 30N",eq[1],

      "in latitude between 30N and 30S and",eq[2],

      "in latitude over 30S. In precentage it is %.2f%%"% eq[3],",",

      "%.2f%%"% eq[4],"and","%.2f%%"% eq[5],"respectively.")
recent_active = volcanoes[(volcanoes["Last Known Eruption"]>='2012 CE') & (volcanoes["Last Known Eruption"]<='2016 CE')]

print(recent_active.shape)

longitudes_vol = recent_active["Longitude"].tolist()

latitudes_vol = recent_active["Latitude"].tolist()



earthquakes_eq["Date"] = pd.to_datetime(earthquakes_eq["Date"])

earthquakes_eq["year"] = earthquakes_eq['Date'].dt.year

last_eq = earthquakes_eq[(earthquakes_eq["year"]>=2012) & (earthquakes_eq["year"]<=2016)]

print(last_eq.shape)



longitudes_eq = last_eq["Longitude"].tolist()

latitudes_eq = last_eq["Latitude"].tolist()



n = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

x,y = n(longitudes_vol,latitudes_vol)

c,d = n(longitudes_eq,latitudes_eq)

fig2 = plt.figure(figsize=(12,10))

plt.title("Volcanoes (red) that were recently active and Earthquakes (green) in the last 5 years")

n.plot(x, y, "o", markersize = 5, color = 'red')

n.plot(c, d, "o", markersize = 3, color = 'green')

n.drawcoastlines()

n.fillcontinents(color='coral',lake_color='aqua')

n.drawmapboundary()

n.drawcountries()

plt.show()
fig_p(volcanoes["Region"])

plt.ylabel("Count")

plt.title("Region with most volcanoes")
plt.figure(figsize=(20,10))

fig_p(volcanoes["Country"])

plt.ylabel("Count")

plt.title("Country with most volcanoes")
from wordcloud import WordCloud



text= ' '

for s, row in volcanoes.iterrows():

    text = " ".join([text,"_".join(row['Country'].strip().split(" "))])

text = text.strip()



plt.figure(figsize=(12,6))

wordcloud = WordCloud(width=600, height=300, max_font_size=60, max_words=20, collocations=False).generate(text)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.title("20 Countries with most recently erupted volcanoes", fontsize=30)

plt.axis("off")

plt.show()
most_vol_region=Series(volcanoes["Region"]).value_counts().sort_index().idxmax(axis=1)

most_vol_country=Series(volcanoes["Country"]).value_counts().sort_index().idxmax(axis=1)



print("The region with most volcanoes is:",most_vol_region)

print("The country with most volcanoes is:",most_vol_country)
fig_p(volcanoes["Activity Evidence"])

plt.ylabel("Count")

plt.title("Common evidanve of activity")
fig_p(volcanoes["Dominant Rock Type"])

plt.ylabel("Count")

plt.title("Dominant Rock Type")
fig_p(volcanoes["Tectonic Setting"])

plt.ylabel("Count")

plt.title("Tectonic Setting")
earthquakes_eq['year'] = earthquakes_eq['Date'].dt.year



plt.figure(figsize=(10,5))

fig_p(earthquakes_eq['year'])

plt.ylabel("Count")

plt.title("Number of earthquakes by year")
most_eq_year=Series(earthquakes_eq['year']).value_counts().sort_index().idxmax(axis=1)

print("The year with most earthquakes is:", most_eq_year)
fig_p(np.around(earthquakes_eq["Magnitude"]))

plt.ylabel("Count")

plt.title("Earthquakes magnitude (round up)")
#still trying to figure out those figures

#plt.scatter(earthquakes_eq["Depth"],earthquakes_eq["Longitude"])

#plt.scatter(earthquakes_eq["Depth"],earthquakes_eq["Latitude"])
earthquakes_nex=pd.DataFrame()

earthquakes_nex=earthquakes[earthquakes['Type']=='Nuclear Explosion']



fig_p(np.around(earthquakes_nex["Magnitude"]))

plt.ylabel("Count")

plt.title("Nuclear Explosions magnitude (round up)")
earthquakes_nex["Date"] = pd.to_datetime(earthquakes_nex["Date"])

earthquakes_nex['year'] = earthquakes_nex['Date'].dt.year

fig_p(earthquakes_nex['year'])

plt.ylabel("Count")

plt.title("Number of Nuclear Explosion by year")