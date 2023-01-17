import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import seaborn as sns

import warnings

%matplotlib inline

warnings.filterwarnings('ignore')



eqdata = pd.read_csv('../input/italy_earthquakes_from_2016-08-24_to_2016-11-30.csv').set_index('Time')
eqdata.head()
eqdata.index = pd.to_datetime(eqdata.index)
mag = eqdata.Magnitude

(n, bins, patches) = plt.hist(mag, bins = 4)

plt.tight_layout()

plt.show()

print(bins)

print(n)
eqdata["Magnitude"].resample('2D').mean().plot()

plt.title("Time Series: Average Magnitude")

plt.ylabel("Magnitude")
print('Highest Magnitude Earthquake:\n','Date/Time:', eqdata.Magnitude.idxmax(),'\n', 'Magnitude:', eqdata.Magnitude.max(),'\n Latitude/Longitude:',eqdata.Latitude[eqdata.Magnitude.idxmax()],',',eqdata.Longitude[eqdata.Magnitude.idxmax()])
def drawmap(df, zoom=1):

    z= (10/3)-(1/3)*zoom

    m = Basemap(projection = 'merc',llcrnrlat=df.Latitude.min()-z, urcrnrlat=df.Latitude.max()+z, llcrnrlon=df.Longitude.min()-z, urcrnrlon=df.Longitude.max()+z)

    x,y = m(list(df.Longitude),list(df.Latitude))

    m.scatter(x,y, c = df.Magnitude, cmap = 'seismic')

    m.colorbar()

    m.drawcoastlines()

    #m.drawstates()

    #m.drawcountries()

    m.bluemarble()

    plt.show()

    plt.clf()
over4 = eqdata[eqdata.Magnitude >=4.25]
drawmap(over4, zoom = -3)

drawmap(over4, zoom = 10)
eqdata.corr()
eqs1030 = eqdata[(eqdata.index >= '2016-10-26') & (eqdata.index <= '2016-11-03') & (eqdata.Magnitude >4)]

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set(title = 'EQs Days before and after OCT.30 6.5 MAG EQ', xlim = [eqs1030.index.min(), eqs1030.index.max()])

ax.scatter(eqs1030.index,eqs1030.Magnitude)

fig.tight_layout()

fig.autofmt_xdate()

plt.show()
