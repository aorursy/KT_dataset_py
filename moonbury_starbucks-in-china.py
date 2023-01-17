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
data = pd.read_csv('../input/directory.csv')

cndata=data[data['Country']=='CN']

cndata.head(2)
import matplotlib.pyplot as plt

from collections import Counter



y=cndata['Brand']

print(Counter(cndata['Brand']))

print(Counter(cndata['Ownership Type']))

#print(Counter(cndata['City']))


plt.figure(figsize=(10,5))



plt.subplot(2,1,1)

c = pd.value_counts(cndata['Ownership Type'],sort=True).sort_index()

c.plot(kind='bar')

plt.title("Ownership")



plt.subplot(2,1,2)

c = pd.value_counts(cndata['State/Province'],sort=True).sort_index()

c.plot(kind='bar')

plt.title("State/Province")
from mpl_toolkits.basemap import Basemap

#get the china basemap

plt.figure(figsize=(10,5))

m = Basemap(llcrnrlon=80.33, llcrnrlat=3.01, urcrnrlon=138.16, urcrnrlat=56.123,

             resolution='h', projection='lcc', lat_0 = 42.5,lon_0=120)

x, y = m(cndata['Longitude'].tolist(),cndata['Latitude'].tolist())

#m.drawcoastlines()

m.shadedrelief()

m.drawcountries()

m.scatter(x,y,3,marker='o',color='b')

plt.title("Starbuck's locations in China")

plt.show()