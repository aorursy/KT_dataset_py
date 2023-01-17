import pandas as pd
import os
import numpy as np
import geopandas as gpd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.collections import PatchCollection
from descartes.patch import PolygonPatch
nb = pd.read_csv("../input/neighbourhoods.csv")
nb.head()
nbgeo = gpd.read_file('../input/neighbourhoods.geojson')
nbgeo.head()
# Price Geometry Distribution 
ls = pd.read_csv("../input/listings_summary.csv")
prices = ls[["id","neighbourhood","price","reviews_per_month"]].groupby(["neighbourhood"]).price.mean()
priceMap = dict(prices)

max_ = max(priceMap.values())
min_ = min(priceMap.values())
def norm_transform(price):
    return (price-min_)/( 250 - min_)#Hard Coding Truncature Here at 250 USD here to see a more meaningful result

###############SET UP LIMITS#########################
mp = nbgeo.geometry.loc[101]
cm = plt.get_cmap('YlOrRd')
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
minx, miny, maxx, maxy = mp.bounds
w, h = maxx - minx, maxy - miny
ax.set_xlim(minx - 20 * w, maxx + 5 * w)
ax.set_ylim(miny - 10 * h, maxy + 8 * h)
ax.set_aspect(1)

###############ADD NEIGHBOURS#########################
for i in range(len(nbgeo.geometry)):
    patches = []
    mp = nbgeo.geometry.loc[i]
    nb = nbgeo.neighbourhood.loc[i]
    try:
        price = priceMap[nb]
    except:
        continue
    for idx, p in enumerate(mp):
        colour = cm(norm_transform(price))
        patches.append(PolygonPatch(p, fc=colour, ec='#555555', lw=0.2, alpha=1., zorder=1))
        ax.add_collection(PatchCollection(patches, match_original=True))

###############DRAW##################################
ax.set_xticks([])
ax.set_yticks([])
plt.title("Price Distributions")
plt.tight_layout()
plt.show()
# Price
price_group = ls[["id","neighbourhood_group","price","reviews_per_month"]].groupby(["neighbourhood_group"]).price.mean()
plt.figure(figsize = (10,10))
plt.bar(range(len(price_group.values)),price_group.values)
plt.xticks(range(len(price_group.values)),list(price_group.index))
plt.title("Average Price")
# reviews_per_month Geometry Distribution 
ls = pd.read_csv("../input/listings_summary.csv")
reviews_per_months = ls[["id","neighbourhood","price","reviews_per_month"]].groupby(["neighbourhood"]).reviews_per_month.mean()
reviews_per_monthMap = dict(reviews_per_months)

max_ = max(reviews_per_monthMap.values())
min_ = min(reviews_per_monthMap.values())
def norm_transform(reviews_per_month):
    return (reviews_per_month-min_)/( 2.5 - min_)#Hard Coding Truncature Here at 2.5  here to see a more meaningful result

###############SET UP LIMITS#########################
mp = nbgeo.geometry.loc[101]
cm = plt.get_cmap('coolwarm')
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
minx, miny, maxx, maxy = mp.bounds
w, h = maxx - minx, maxy - miny
ax.set_xlim(minx - 20 * w, maxx + 5 * w)
ax.set_ylim(miny - 10 * h, maxy + 8 * h)
ax.set_aspect(1)

###############ADD NEIGHBOURS#########################
for i in range(len(nbgeo.geometry)):
    patches = []
    mp = nbgeo.geometry.loc[i]
    nb = nbgeo.neighbourhood.loc[i]
    try:
        reviews_per_month = reviews_per_monthMap[nb]
    except:
        continue
    for idx, p in enumerate(mp):
        colour = cm(norm_transform(reviews_per_month))
        patches.append(PolygonPatch(p, fc=colour, ec='#555555', lw=0.2, alpha=1., zorder=1))
        ax.add_collection(PatchCollection(patches, match_original=True))

###############DRAW##################################
ax.set_xticks([])
ax.set_yticks([])
plt.title("Average number of reviews_per_month Distributions")
plt.tight_layout()
plt.show()
# Number of Reiviews per Month
reviews_per_months_group = ls[["id","neighbourhood_group","price","reviews_per_month"]].groupby(["neighbourhood_group"]).reviews_per_month.mean()
plt.figure(figsize = (10,10))
plt.bar(range(len(reviews_per_months_group.values)),reviews_per_months_group.values)
plt.xticks(range(len(reviews_per_months_group.values)),list(reviews_per_months_group.index))
plt.title("Avearge Number of Reiviews per Month")
# Listing ID count Geometry Distribution 
ls = pd.read_csv("../input/listings_summary.csv")
counts = ls[["id","neighbourhood","price","reviews_per_month"]].groupby(["neighbourhood"]).id.count()
countsMap = dict(counts.apply(np.log))

max_ = max(countsMap.values())
min_ = min(countsMap.values())
def norm_transform(count):
    return (count-min_)/( max_ - min_)#Hard Coding Truncature Here at 250 USD here to see a more meaningful result

###############SET UP LIMITS#########################
mp = nbgeo.geometry.loc[101]
cm = plt.get_cmap('Reds')
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
minx, miny, maxx, maxy = mp.bounds
w, h = maxx - minx, maxy - miny
ax.set_xlim(minx - 20 * w, maxx + 5 * w)
ax.set_ylim(miny - 10 * h, maxy + 8 * h)
ax.set_aspect(1)

###############ADD NEIGHBOURS#########################
for i in range(len(nbgeo.geometry)):
    patches = []
    mp = nbgeo.geometry.loc[i]
    nb = nbgeo.neighbourhood.loc[i]
    try:
        count = countsMap[nb]
    except:
        continue
    for idx, p in enumerate(mp):
        colour = cm(norm_transform(count))
        patches.append(PolygonPatch(p, fc=colour, ec='#555555', lw=0.2, alpha=1., zorder=1))
        ax.add_collection(PatchCollection(patches, match_original=True))

###############DRAW##################################
ax.set_xticks([])
ax.set_yticks([])
plt.title("Number of ListingId Distributions")
plt.tight_layout()
plt.show()
# Listing ID count
count_group = ls[["id","neighbourhood_group","price","reviews_per_month"]].groupby(["neighbourhood_group"]).id.count()
plt.figure(figsize = (10,10))
plt.bar(range(len(count_group.values)),count_group.values)
plt.xticks(range(len(count_group.values)),list(count_group.index))
plt.title("Number of ListingIDs")