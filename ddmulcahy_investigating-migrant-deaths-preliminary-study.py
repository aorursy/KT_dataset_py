import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

from wordcloud import WordCloud, STOPWORDS

%matplotlib inline
raw_data = pd.read_csv('../input/MissingMigrantsProject.csv',encoding = "ISO-8859-1")
raw_data.info()
raw_data
raw_data['cause_of_death'].unique()
deaths = raw_data['cause_of_death']

deaths = deaths.dropna()
wordcloud_deaths = WordCloud(

                          background_color='white',

                          stopwords=set(STOPWORDS),

                          max_words=250,

                          max_font_size=40, 

                          random_state=1705

                         ).generate(str(deaths))

def cloud_plot(wordcloud):

    fig = plt.figure(1, figsize=(20,15))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
cloud_plot(wordcloud_deaths)
lat = raw_data['lat'][:]

lon = raw_data['lon'][:]

lat = lat.dropna()

lon = lon.dropna()

lat = np.array(lat)

lon = np.array(lon)



fig=plt.figure()

ax=fig.add_axes([1.0,1.0,2.8,2.8])

map = Basemap(llcrnrlon=-180.,llcrnrlat=-60.,urcrnrlon=180.,urcrnrlat=80.,\

            rsphere=(6378137.00,6356752.3142),\

            resolution='l',projection='merc',\

            lat_0=40.,lon_0=-20.,lat_ts=20.)

map.drawcoastlines()

map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])

map.drawmeridians(np.arange(map.lonmin,map.lonmax+30,60),labels=[0,0,0,1])

x, y = map(lon,lat)

map.scatter(x,y,3,marker='o',color='r')

ax.set_title('Refugee deaths across the world')

plt.show()