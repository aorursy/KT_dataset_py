import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

from mpl_toolkits.basemap import Basemap

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv",usecols = ['ID','age','sex','province','country','latitude','longitude'])

df.dropna(inplace=True)
df.info()
plt.figure(figsize=(20,10))

m=Basemap(llcrnrlon=-160, llcrnrlat=-75,urcrnrlon=160,urcrnrlat=80)

m.shadedrelief()



m.plot(df['longitude'],df['latitude'], linestyle='none', marker="o", markersize=5, alpha=0.6, c="orange", markeredgecolor="red")
df.groupby(['country']).size()