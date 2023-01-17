import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
%matplotlib inline
df_india = gpd.read_file('https://raw.githubusercontent.com/geohacker/india/master/taluk/india_taluk.geojson')
df_maharashtra = df_india[df_india['NAME_1']== 'Maharashtra']
df_jalgaon = df_maharashtra[df_maharashtra['NAME_2']=='Jalgaon']
df_jamner = df_jalgaon[df_jalgaon['NAME_3']=='Jamner']
plt.rcParams['figure.figsize'] = (30, 10)
ax = df_india.plot(color='blue')
plt.rcParams['figure.figsize'] = (30, 10)
ax = df_maharashtra.plot(color='blue')
plt.rcParams['figure.figsize'] = (30, 10)
ax = df_jalgaon.plot(color='blue')
plt.rcParams['figure.figsize'] = (30, 10)
ax = df_jamner.plot(color='blue')
plt.rcParams['figure.figsize'] = (30, 10)
ax = df_maharashtra[df_maharashtra['NAME_2']=='Latur'].plot(color='blue')
