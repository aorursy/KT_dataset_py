import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import numpy as np

import folium

from folium import plugins
gvd = pd.read_excel('/kaggle/input/1st-task/gun_violence_data_2013_2018.xlsx', sheet_name = 'Arkusz2')

gvd
mkis = gvd.groupby('state')['n_killed'].sum()

mkis = mkis.rename_axis("state")
population = [4779736,710231,6392017,2915918,37253956,5029196,3574097,897934,601723,18801310,9687653,1360301,1567582,12830632,6483802,3046355,2853118,4339367,4533372,1328361,5773552,6547629,9883640,5303925,2967297,5988927,989415,1826341,2700551,1316470,8791894,2059179,19378102,9535483,672591,11536504,3751351,3831074,12702379,1052567,4625364,814180,6346105,25145561,2763885,625741,8001024,6724540,1852994,5686986,563626,]

popdens = [95.8,1.3,60.1,57.2,251.0,52.6,741.2,484.1,11011.0,375.9,176.4,222.9,20.0,231.4,184.6,55.9,35.6,111.4,107.2,43.1,614.5,866.6,174.7,69.0,63.8,88.3,7.1,24.7,26.3,148.4,1207.8,17.2,419.3,206.2,11.0,283.6,57.0,42.0,285.7,1010.8,162.6,11.3,160.1,104.9,36.5,67.7,211.7,107.8,76.6,106.3,6.0,]
kpr = pd.DataFrame(mkis)

kpr.reset_index(inplace=True)
kpr['population'] = population

kpr['population density'] = popdens

ratio = kpr.n_killed / kpr.population * 100000

kpr['murders per 100,000 citizens'] = round(ratio, 1)

kpr.to_csv('project.csv')

kpr.sort_values('murders per 100,000 citizens', ascending=False).reset_index(drop=True).head(5)
y = kpr['murders per 100,000 citizens']

z = kpr['population']

n = kpr['state']



fig, ax = plt.subplots(figsize=(17,14))

ax.scatter(z, y)

ax.set_xscale('log')

ax.set_xlabel("population")

ax.set_ylabel("murders per 100,000 citizens")



for i, txt in enumerate(n):

    ax.annotate(txt, (z[i], y[i]))
y = kpr['murders per 100,000 citizens']

z = kpr['population density']

n = kpr['state']



fig, ax = plt.subplots(figsize=(17,22))

ax.scatter(z, y)

ax.set_xscale('log')

ax.set_xlabel("popdens")

ax.set_ylabel("murders per 100,000 citizens")



for i, txt in enumerate(n):

    ax.annotate(txt, (z[i], y[i]))
kpr.reset_index(drop=True).sort_values(by = 'murders per 100,000 citizens', ascending = False).iloc[np.r_[0:4, 47:51]].plot(

    x='state', y='murders per 100,000 citizens', kind = 'bar',figsize=(15,5), color = 'red', alpha = 0.7, legend=False);
us_gvd_map = folium.Map(location=[54.58, -103.46], zoom_start=3)



accidents = plugins.MarkerCluster().add_to(us_gvd_map)



gvd.dropna(subset = ["latitude"], inplace=True)



for lat, lng, label in zip(gvd.latitude, gvd.longitude, gvd.n_killed.astype(str)):

    if label!='0':

        folium.Marker(

            location=[lat, lng],

            icon=None,

            popup=label,

        ).add_to(accidents)



us_gvd_map