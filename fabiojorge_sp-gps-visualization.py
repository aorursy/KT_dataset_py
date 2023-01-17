import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("../input/gps-points-mobile-phone-sp/SaoPauloStateGPSPointsMobilePhone.csv")
pd.set_option("display.max_rows", 5)
data.head()
data.size
data.describe
# FILTER BY SÃO PAULO STATE 
X = data[data.state == "SP"]

#SORT BY ID, DAY, TIME
X = X.sort_values(by=['id', 'beginDay', 'beginTime'])
X.loc[:, ['lat', 'lng']]
#LOAD VISUALIZATION LIB
import matplotlib.pyplot as plt
import matplotlib
import json
from descartes import PolygonPatch

matplotlib.rcParams['figure.figsize'] = [72.0, 48.0]
#OPEN GEOJSON FROM SÃO PAULO SATE'S CITIES
with open("../input/geojs35munsp/geojs-35-mun.json", 'r', encoding='ISO-8859-1') as json_file:
    json_data = json.load(json_file)
#LOAD CITIES INTO THE PLOT
fig1 = plt.figure(1)
ax = fig1.gca()
# fig, ax = plt.subplots()

for poly in json_data["features"]:
    coords = poly["geometry"]["coordinates"]
    x = [i for i,j in coords[0]]
    y = [j for i,j in coords[0]]
    ax.plot(x,y, color="b")

plt.plot(X.lng, X.lat, 'o', color=tuple([1,0,0]), markeredgecolor='k', markersize=6)
plt.show()