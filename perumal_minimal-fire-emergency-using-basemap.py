# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from mpl_toolkits.basemap import Basemap

%matplotlib inline

df = pd.read_csv('../input/911.csv')

df.head()

# Any results you write to the current directory are saved as output.
# Fire Emergency Data

FireEmergency = df[df['title'].str.contains('Fire:')]

print(FireEmergency['title'].unique()) 
GasODOR = df[df['title'].str.contains('Fire: GAS-ODOR/LEAK')]  

             

LowerLattitude=FireEmergency['lat'].min()

UpperLattitude=FireEmergency['lat'].max()

LowerLontitude=FireEmergency['lng'].min()

UpperLontitude=FireEmergency['lng'].max()

           

plt.figure(figsize=(20,10))

m = Basemap(projection='mill', llcrnrlat=LowerLattitude, urcrnrlat=UpperLattitude, 

            llcrnrlon=LowerLontitude, urcrnrlon=UpperLontitude, resolution='h', epsg=4269)

x, y = m(tuple(GasODOR.lng[(GasODOR.lng.isnull()==False)]), \

         tuple(GasODOR.lat[(GasODOR.lat.isnull() == False)]))             

#m.arcgisimage(service="NatGeo_World_Map", xpixels=9000, verbose=True)

#m.plot(x,y,'ro',markersize=1, alpha=.3, color='R' )

#plt.show()
CarbonMonoxideFire = df[df['title'].str.contains('Fire: CARBON MONOXIDE DETECTOR')]   

              

plt.figure(figsize=(100,50))

m = Basemap(projection='mill', llcrnrlat=LowerLattitude, urcrnrlat=UpperLattitude, 

            llcrnrlon=LowerLontitude, urcrnrlon=UpperLontitude, resolution='h', epsg=4269)

x, y = m(tuple(CarbonMonoxideFire.lng[(CarbonMonoxideFire.lng.isnull()==False)]), \

         tuple(CarbonMonoxideFire.lat[(CarbonMonoxideFire.lat.isnull() == False)]))             

#m.arcgisimage(service="NatGeo_World_Map", xpixels=3000, verbose=True)

#m.plot(x,y,'ro',markersize=10, alpha=.3, color='R' )

#plt.show()