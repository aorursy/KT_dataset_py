import pandas as pd

plaques = pd.read_csv("../input/blue-plaques/open-plaques-all-2017-06-19.csv", index_col=0)

pd.set_option('max_columns', None)

plaques.head(2)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")



f, axarr = plt.subplots(2, 2, figsize=(12, 11))

f.subplots_adjust(hspace=0.75)

plt.suptitle('Plaque Locations, Times Erected, and Colors', fontsize=18)



plaques.country.value_counts().head(10).plot.bar(ax=axarr[0][0])

axarr[0][0].set_title("Plaques per Country (n=10)")



plaques.area.value_counts().head(20).plot.bar(ax=axarr[0][1])

axarr[0][1].set_title("Plaques per City (n=20)")



plaques.erected.value_counts().sort_index().tail(100).plot.line(ax=axarr[1][0])

axarr[1][0].set_title("Plaques erected over Time (t > 1910)")



plaques.colour.value_counts().head(10).plot.bar(ax=axarr[1][1], color='darkgray')

axarr[1][1].set_title("Plaque colors (n=10)")
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")



f, axarr = plt.subplots(1, 2, figsize=(12, 4))

plaques['geolocated?'].value_counts(dropna=False).sort_index()[::-1].plot.bar(ax=axarr[0])

axarr[0].set_title("geolocated?")



plaques['photographed?'].value_counts(dropna=False).sort_index()[::-1].plot.bar(ax=axarr[1])

axarr[1].set_title("photographed?")

axarr[1].set_ylim([0, 35000])
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")



f, axarr = plt.subplots(2, 2, figsize=(12, 11))

f.subplots_adjust(hspace=0.75)

# plt.suptitle('Plaque Locations, Times Erected, and Colors', fontsize=18)



plaques.lead_subject_type.value_counts(dropna=False).plot.bar(ax=axarr[0][0])

axarr[0][0].set_title("Subject Kind")



plaques.number_of_subjects.value_counts(dropna=False).sort_index().plot.bar(ax=axarr[0][1])

axarr[0][1].set_title("# of Subjects")



plaques.lead_subject_born_in.plot.hist(ax=axarr[1][0], bins=200)

axarr[1][0].set_title("Lead Subject Born In")

axarr[1][0].set_xlim([1500, 2020])



plaques.lead_subject_died_in.plot.hist(ax=axarr[1][1], bins=200)

axarr[1][1].set_title("Lead Subject Died In")

axarr[1][1].set_xlim([1500, 2020])
pd.set_option('max_colwidth',1000)

plaques.query('lead_subject_type == "animal"')[['inscription', 'country']]
pd.reset_option('max_colwidth')
(plaques.lead_subject_died_in - plaques.lead_subject_born_in).where(lambda v: (v < 100) & (0 < v)).dropna().plot.hist(bins=30)

plt.suptitle('Subject Lifetimes')
from wordcloud import WordCloud

w = WordCloud(width=800, height=400)

w.generate(" ".join(list(plaques.inscription.values.astype(str))))
w.to_image()
latlongs = plaques[['latitude', 'longitude']].dropna()

from shapely.geometry import Point

points = latlongs.apply(lambda srs: Point(srs.longitude, srs.latitude), axis='columns')

import geopandas as gpd

gplaques = gpd.GeoDataFrame(plaques, geometry=points)
world_countres = gpd.read_file("../input/countries-shape-files/ne_10m_admin_0_countries.shp")
import geoplot as gplt

import geoplot.crs as gcrs



ax = gplt.polyplot(world_countres, projection=gcrs.PlateCarree(), linewidth=0.5, 

                   figsize=(14, 8))

gplt.pointplot(gplaques.loc[pd.notnull(gplaques.geometry)], projection=gcrs.PlateCarree(), 

               edgecolor='black', alpha=1, s=69,

               ax=ax)
ax = gplt.polyplot(world_countres, projection=gcrs.PlateCarree(), linewidth=0.5, 

                   figsize=(14, 8))

gplt.kdeplot(gplaques.loc[pd.notnull(gplaques.geometry)], projection=gcrs.PlateCarree(), 

             ax=ax, clipped=True)