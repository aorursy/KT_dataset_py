# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install git+https://github.com/python-visualization/branca

!pip install git+https://github.com/sknzl/folium@update-css-url-to-https
import matplotlib.pyplot as plt

import folium

from folium import plugins
data = pd.read_csv('/kaggle/input/fires-from-space-australia-and-new-zeland/fire_nrt_V1_96617.csv')

sample_data = data.iloc[np.random.choice(data.index, 1000)]
data.head()
fig, axes = plt.subplots(2,2, figsize =(20,15), subplot_kw={'xticks': (), 'yticks': ()})

axes[0,0].scatter(data.longitude,data.latitude)

axes[0,0].set_title("Default scatter")

axes[0,1].scatter(data.longitude,data.latitude, alpha=0.01, marker=".")

axes[0,1].set_title("Scatter with dot marker and alpha 0.01")

img = axes[1,0].hexbin(data.longitude,data.latitude, bins='log')

axes[1,0].set_title("Hexbin map with color legend and no axis")

axes[1,1].scatter(sample_data.longitude, sample_data.latitude)

axes[1,1].set_title("Default Scatter on 1K subsampled data")

for ax in axes.ravel():

    ax.set_xlabel('Latitude')

    ax.set_ylabel('Longitude')

plt.tight_layout()

fig.colorbar(img, ax=axes[1,0])

plt.show()
#Create a map

f = folium.Figure(width=1000, height=500)

center_lat = -24.003249 

center_long = 133.737310

m = folium.Map(location=[center_lat,center_long], control_scale=True, zoom_start=4,width=750, height=500,zoom_control=True).add_to(f)

for i in range(0,sample_data.shape[0]):    

    location=[sample_data.iloc[i]['latitude'], sample_data.iloc[i]['longitude']]

    folium.CircleMarker(location,radius=1,color='red').add_to(m)

m
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.hist(data.bright_ti4,bins='auto');

ax.set_title("Histogram of Brightness Temperatures I4");

ax.set_xlabel("Brightness Temperature I4");

ax.set_ylabel("Counts");
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.hist(data.bright_ti4,bins=200);

ax.set_xlim(290,360);

ax.set_ylim(5000,1.55e4);

ax.set_title("Histogram of Brightness Temperatures I4");

ax.set_xlabel("Brightness Temperature I4");

ax.set_ylabel("Counts");
mask = data['bright_ti4'] >= 330

sat_df = data[mask]

unsat_df = data[~mask]
fig, axes = plt.subplots(1,2, figsize =(20,10), subplot_kw={'xticks': (), 'yticks': ()})

img = axes[0].scatter(sat_df.longitude,sat_df.latitude, alpha=0.01, color='red')

axes[0].set_title("Scatter map for Saturated Brightness I4")

img = axes[1].scatter(unsat_df.longitude,unsat_df.latitude, alpha=0.01)

axes[1].set_title("Scatter map for UnSaturated Brightness I4")



for ax in axes.ravel():

    ax.set_xlabel('Longitude')

    ax.set_ylabel('Latitude')

plt.tight_layout()

#fig.colorbar(img, ax=axes[1])

plt.show()
fig, axes = plt.subplots(1,2, figsize =(20,10), subplot_kw={'xticks': (), 'yticks': ()})

img = axes[0].scatter(sat_df.longitude,sat_df.latitude, alpha=0.008, color='red')

axes[0].set_title("Scatter map for Saturated Brightness I4")

img = axes[1].scatter(unsat_df.longitude,unsat_df.latitude, alpha=0.008)

axes[1].set_title("Scatter map for UnSaturated Brightness I4")



for ax in axes.ravel():

    ax.set_xlabel('Longitude')

    ax.set_ylabel('Latitude')

plt.tight_layout()

#fig.colorbar(img, ax=axes[1])

plt.show()