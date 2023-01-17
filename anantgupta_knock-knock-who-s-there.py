# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


#!head -5 ../input/complete.csv

data=pd.read_csv('../input/complete.csv',sep=',',error_bad_lines=False,verbose=False)

data['latitude']=data['latitude'].map(lambda x: x if type(x) in (float,int) else 999)

data['longitude']=data['longitude'].map(lambda x: x if type(x) in (float,int) else 999)

data=data[data['latitude'] != 999]

data=data[data['longitude'] != 999]



# What areas of the country are most likely to have UFO sightings?

# Are there any trends in UFO sightings over time? Do they tend to be clustered or seasonal?

# Do clusters of UFO sightings correlate with landmarks, such as airports or government research centers?

# What are the most common UFO descriptions?



import time, calendar, datetime, numpy

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt



plt.figure(figsize=(20,10))

# Lambert Conformal Conic map.

#m = Basemap(llcrnrlon=-100.,llcrnrlat=0.,urcrnrlon=-20.,urcrnrlat=57.,

#            projection='lcc',lat_1=10.,lat_2=50.,lon_0=-50.,

#            resolution ='l',area_thresh=10000.)

m = Basemap(width=12000000,height=9000000,

            rsphere=(6378137.00,6356752.3142),\

            resolution='l',area_thresh=1000.,projection='lcc',\

            lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)

m.bluemarble()

#m = Basemap(projection='hammer',lon_0=-100,lat_0=45)

x, y = m(data['longitude'].values,data['latitude'].values)

m.drawmapboundary(fill_color='#99ffff')

m.fillcontinents(color='#cc9966',lake_color='#99ffff')

m.scatter(x,y,20,marker='o',color='red')



plt.title('Plotting the location of all UFO sightings',fontsize=12)

plt.show()
#Are there any trends in UFO sightings over time? Do they tend to be clustered or seasonal?

#from datetime import datetime

#data['hour']=data['datetime'].map(lambda x:datetime.strptime(x,"%m/%d/%Y %H:%M"))

#date['hour']

#datetime.strptime('7/23/1998 24:00',"%m/%d/%Y %H:%M")

#datestr='3/1/2014 9:55'

#datetime.strptime(datestr,"%m/%d/%Y %H:%M")