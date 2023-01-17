# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



%matplotlib inline

import numpy as np

import pandas as pd

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
df = pd.read_csv('../input/nyc-elevators.csv')

print(df.columns)

df.head(n=5)
#Exploratory

fig, axs = plt.subplots(1,2)

#Device Type distribution

df['Device Type'].value_counts().plot(title = 'Device Types', kind='bar',ax=axs[0])



#Device Status Distribution

df['Device Status'] = df['Device Status'].astype(str)

df['Device Status'].value_counts().plot(title = 'Device Status',kind='bar', ax=axs[1])

plt.tight_layout(pad=1, w_pad=1, h_pad=3.0)
#Geographic distribution of elevators!





lats = df['LATITUDE'].values

longs = df['LONGITUDE'].values



#Used this site to easily get bounding box coordinates for the Basemap object:

#http://boundingbox.klokantech.com/

#westlimit=-74.2568; southlimit=40.4915; eastlimit=-73.6961; northlimit=40.9191

mfig = Basemap(llcrnrlon=-74.2568,

            llcrnrlat=40.4915,

            urcrnrlon=-73.6961,

            urcrnrlat=40.9191,

              resolution = 'f')

mfig.drawcoastlines()

mfig.drawmapboundary(fill_color='aqua')

mfig.fillcontinents(color='coral',lake_color='aqua')



long, lat = mfig(longs, lats)

mfig.scatter(long, lat, s= 0.1, marker = 'o', color = 'k',zorder=10)

#Mapping Freight Elevators



freight_lats = df[df['Device Type'] == 'Freight (F)']['LATITUDE'].values

freight_longs = df[df['Device Type'] == 'Freight (F)']['LONGITUDE'].values



mfig = Basemap(llcrnrlon=-74.2568,

            llcrnrlat=40.4915,

            urcrnrlon=-73.6961,

            urcrnrlat=40.9191,

              resolution = 'f')

mfig.drawcoastlines()

mfig.drawmapboundary(fill_color='aqua')

mfig.fillcontinents(color='coral',lake_color='aqua')





flong, flat = mfig(freight_longs, freight_lats)

mfig.scatter(flong, flat, s= 0.1, marker = 'o', color = 'k',zorder=10)