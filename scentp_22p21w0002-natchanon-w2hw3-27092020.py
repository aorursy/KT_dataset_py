import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
air = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
air.info()
# findind coordinate of the map

# max(air["longitude"]), min(air["longitude"]), max(air["latitude"]), min(air["latitude"])
air.plot(x="longitude", y="latitude", style=".", figsize=(10, 10))

plt.title("NY map")

plt.ylabel("latitude")

img = plt.imread("/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png", 0)

plt.imshow(img, extent=[-74.25, -73.685, 40.49, 40.925])

plt.show()
# change room_type to numerical code in room_code

room_type = {"Entire home/apt":1, "Private room":2, "Shared room":3}

room_code = []

for i in range(len(air)):

    room_code.append(room_type[air["room_type"][i]])

air.insert(loc=9, column="room_code", value=room_code, allow_duplicates=True)
air.head()
# feature select

cols = ["room_code", "price", "minimum_nights", "number_of_reviews", "availability_365"]
air[cols].info()
from sklearn import preprocessing

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch # draw dendrogram

import seaborn as sns

import random



air = air[:500] # too big data can't run at the same time



pt = preprocessing.PowerTransformer() # support only positive value

mat = pt.fit_transform(air[cols])

# mat[:5].round(4)



std=pd.DataFrame(mat, columns=cols)

std.head()
# non-standardize

air[cols].hist(layout=(1, len(cols)), figsize=(2*len(cols), 2));
# standradize

std[cols].hist(layout=(1, len(cols)), figsize=(2*len(cols), 2), color="red");
fig, ax=plt.subplots(figsize=(20, 7))

dg = sch.dendrogram(sch.linkage(std, method='ward'), ax=ax, labels=air['name'].values)
# without standardize

sns.clustermap(air[cols], col_cluster=False, cmap="Reds")
# with starndardize

sns.clustermap(std, col_cluster=False, cmap="Reds")