# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

from mpl_toolkits.basemap import Basemap

%matplotlib inline

#mapdata=pd.read_csv('C://Assignment/map_data.csv',sep=',',names=['Longitude','Latitude'])



quakedata = pd.read_csv('../input/database.csv')

nucleardata=quakedata[quakedata.Type=='Nuclear Explosion']

#toget only earthquake data

quakedata=quakedata[quakedata.Type=='Earthquake']



#filter to get india data

indiadata= quakedata[(quakedata.Latitude>6.00) & (quakedata.Latitude<36.00)&(quakedata.Longitude>69.00) & (quakedata.Longitude<98.00)]

indiadata=indiadata[indiadata.Type=='Earthquake']

#indiadata=indiadata[indiadata.Status=='Reviewed']



#print indiadata

m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

x,y=m(quakedata.Longitude.tolist(),quakedata.Latitude.tolist())

m.scatter(x,y,3,marker='o',color='Blue')

m.drawcoastlines()

m.fillcontinents(color='White',lake_color='aqua')

m.drawmapboundary(fill_color='aqua')

plt.title("Locations of earthquakes since 1965")

#plt.savefig('EartquakeLoc.png', format='png', dpi=900)











#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
indiadata= quakedata[(quakedata.Latitude>6.00) & (quakedata.Latitude<36.00)&(quakedata.Longitude>69.00) & (quakedata.Longitude<98.00)]

indiadata=indiadata[indiadata.Type=='Earthquake']

m = Basemap(projection='merc',llcrnrlat=5,urcrnrlat=40,llcrnrlon=65,urcrnrlon=100,lat_ts=20,resolution='c')

x,y=m(indiadata.Longitude.tolist(),indiadata.Latitude.tolist())

m.scatter(x,y,3,marker='x',color='Blue')

m.drawcoastlines()

m.fillcontinents(color='White',lake_color='aqua')

m.drawmapboundary(fill_color='aqua')

#plt.scatter(indiadata.Longitude,indiadata.Latitude)

#plt.savefig('indiaeartquake.png', format='png', dpi=900)

#plt.show()
IntenseQuake=quakedata[quakedata.Magnitude>8]

area = np.pi * (IntenseQuake.Magnitude)**2

plt.scatter(IntenseQuake.Longitude,IntenseQuake.Latitude,s=area,alpha=0.5)

#plt.savefig('intenseearthquake.png', format='png', dpi=900)

plt.show()
longitudelabels=[

'(-180, -170]',

'(-170, -160]',

'(-160, -150]',

'(-150, -140]',

'(-140, -130]',

'(-130, -120]',

'(-120, -110]',

'(-110, -100]',

'(-100, -90]',

'(-90, -80]',

'(-80, -70]',

'(-70, -60]',

'(-60, -50]',

'(-50, -40]',

'(-40, -30]',

'(-30, -20]',

'(-20, -10]',

'(-10, 0]',

'(0, 10]',

'(10, 20]',

'(20, 30]',

'(30, 40]',

'(40, 50]',

'(50, 60]',

'(60, 70]',

'(70, 80]',

'(80, 90]',

'(90, 100]',

'(100, 110]',

'(110, 120]',

'(120, 130]',

'(130, 140]',

'(140, 150]',

'(150, 160]',

'(160, 170]',

'(170, 180]'

]



Latitudelabels=[

'(-90, -80]',

'(-80, -70]',

'(-70, -60]',

'(-60, -50]',

'(-50, -40]',

'(-40, -30]',

'(-30, -20]',

'(-20, -10]',

'(-10, 0]',

'(0, 10]',

'(10, 20]',

'(20, 30]',

'(30, 40]',

'(40, 50]',

'(50, 60]',

'(60, 70]',

'(70, 80]',

'(80, 90]'

]



quakedata['LongitudeGrp']=pd.cut(quakedata.Longitude,np.arange(-180,181,10),labels=longitudelabels)

quakedata['LatitudeGrp']=pd.cut(quakedata.Latitude,np.arange(-90,91,10),labels=Latitudelabels.reverse())

print( quakedata[['Date','Longitude','Latitude','LatitudeGrp','LongitudeGrp',]].head(100))



plt.hist(quakedata.Longitude,bins=36)

#plt.savefig('Logitude.png', format='png', dpi=900)

plt.show()

print( quakedata.groupby('LongitudeGrp').agg({'Magnitude':[np.size,max,np.mean]}))
plt.hist(quakedata.Latitude,bins=18)

#plt.savefig('Latitude.png', format='png', dpi=900)

plt.show()

print( quakedata.groupby('LatitudeGrp').agg({'Magnitude':[np.size,max,np.mean]}))
heatmapdata=quakedata.groupby(['LongitudeGrp','LatitudeGrp']).size().reset_index().rename(columns={0:'Count'})

heatmapdata=heatmapdata.pivot(index='LatitudeGrp',columns='LongitudeGrp',values='Count')

df=pd.DataFrame(columns=longitudelabels)

df.loc['(-90, -80]']=np.NaN

heatmapdata=df.append(heatmapdata)

heatmapdata=heatmapdata.reindex(index=heatmapdata.index[::-1])

plt.imshow(heatmapdata,cmap=plt.cm.Blues,interpolation='none')

plt.xticks(np.arange(36),longitudelabels,rotation='vertical')

#LatitudelabelsR=Latitudelabels[]

plt.yticks(np.arange(18),Latitudelabels)

#plt.savefig('heatmap.png', format='png', dpi=900)

plt.show()
quakedata['Date']=pd.to_datetime(quakedata.Date)

IntenseQuake['Date']=pd.to_datetime(IntenseQuake.Date)



print (quakedata.groupby(quakedata.Date.dt.year).agg({'Magnitude':[np.size,max,np.mean]}))

print (IntenseQuake.groupby(IntenseQuake.Date.dt.year).agg({'Magnitude':[np.size,max,np.mean]}))

print (quakedata.groupby(quakedata.Date.dt.year).size())

quakedata.groupby(quakedata.Date.dt.year).size().plot(kind='line')
print (quakedata.groupby(quakedata.Date.dt.month).size())

quakedata.groupby(quakedata.Date.dt.month).size().plot(kind='line')
