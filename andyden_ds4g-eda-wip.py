import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium

import seaborn as sns

import matplotlib as pl

%matplotlib inline

import json

import os
# Let's import plant data

gp = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')



#Flatten geo column

gp_f = gp.join(pd.io.json.json_normalize(gp[".geo"].map(json.loads).tolist()).add_prefix("geo.")).drop([".geo"], axis=1)

gp_f['longitude'] = gp_f['geo.coordinates'].map(lambda x: x[0])

gp_f['latitude'] = gp_f['geo.coordinates'].map(lambda x: x[1])

gp_f
sw = [gp_f['latitude'].min(), gp_f['longitude'].min()]

ne = [gp_f['latitude'].max(), gp_f['longitude'].max()]



m = folium.Map(tiles="Stamen Terrain")

m.fit_bounds([sw, ne])



for i in range(0, len(gp_f.index)):

    folium.Marker([gp_f.iloc[i]['latitude'], gp_f.iloc[i]['longitude']], popup=gp_f.iloc[i]['primary_fuel']).add_to(m)

m
print("Total capacity:", gp_f['capacity_mw'].sum(), "MW/h")

print("Estimated cumulative generation:", gp_f['estimated_generation_gwh'].sum(), "GW/h")
plot = gp_f['primary_fuel'].value_counts().plot.pie(autopct='%1.0f%%')

plot.set_title("Sources by # of plants")

pl.pyplot.show()



plot = gp_f['primary_fuel'].value_counts().plot.bar()

plot.set_title("# of plants by source")

pl.pyplot.show()



print("Top 3:", gp_f[gp_f['primary_fuel'].isin(['Hydro', 'Gas', 'Solar'])]['capacity_mw'].sum() * 100 / gp_f['capacity_mw'].sum())

print("Renewables:", gp_f[gp_f['primary_fuel'].isin(['Hydro', 'Wind', 'Solar'])]['capacity_mw'].sum() * 100/ gp_f['capacity_mw'].sum())
plot = gp_f.groupby('primary_fuel')['capacity_mw'].sum().plot.pie(autopct='%1.0f%%')

plot = gp_f.groupby('primary_fuel')['estimated_generation_gwh'].sum().plot.pie(autopct='%1.0f%%')

plot.set_title("Historical generation by source")

plot.set_ylabel("")

pl.pyplot.show()
gp_f['owner'].value_counts().plot.bar()
hist = gp_f['capacity_mw'].plot.hist()

hist.set_xlabel("MW")

pl.pyplot.show()
from datetime import datetime

import os



files=[]

for dirname, _, filenames in os.walk('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))
import rasterio as rio

from matplotlib import pyplot as plt



pr_bounds = [[18.6,-67.3,],[17.9,-65.2]]

pr_center = [np.mean(x) for x in zip(*pr_bounds)]

print(pr_center)



def overlay_on_pr(file_name,band_layer):

    band = rio.open(file_name).read(band_layer)

    print(band)

    m = folium.Map(pr_center, zoom_start=8)

    folium.raster_layers.ImageOverlay(

        image=band,

        bounds=pr_bounds,

        colormap=lambda x: (1, 0, 0, x)

    ).add_to(m)

    return m



def plot_scaled(file_name):

    vmin, vmax = np.nanpercentile(file_name, (5,95))  # 5-95% stretch

    img_plt = plt.imshow(file_name, cmap='Purples_r', vmin=vmin, vmax=vmax)

    plt.show()

    

image_band = rio.open(files[10]).read(1)

plot_scaled(image_band)

overlay_on_pr(files[10], 1)
from dateutil.parser import parse

# Process and create DataFrame

list_no2 = []

print("There are", len(files), "NO2 Sentinel measurements.")

for file in files:

    no2 = dict()

    filename = os.path.splitext(os.path.basename(file))[0]

    no2['start_ts'] = parse(filename.split('_')[-2])

    no2['end_ts'] = parse(filename.split('_')[-1]) 

    no2['data'] = rio.open(file).read(1)

    no2['sum_no2'] = np.sum(no2['data'])

    no2['avg_no2'] = np.mean(no2['data'])

    no2['month'] = no2['start_ts'].strftime('%Y-%m')

    list_no2.append(no2)

no2_df = pd.DataFrame(list_no2)

print("Processing files and loading into DF finished.")
plt.figure(figsize=[8,4])

chart = sns.lineplot('month', 'avg_no2', data=no2_df)

plt.setp(chart.get_xticklabels(), rotation=45)

chart
no2_df['avg_no2'].mean() / (gp_f['capacity_mw'].sum() / 365)