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
import pandas as pd

import numpy as np

from datetime import datetime

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import sklearn as skl

from sklearn.ensemble import RandomForestClassifier
pmd = pd.read_csv('../input/300k.csv',low_memory=False)
lats = pmd.latitude.values

lons = pmd.longitude.values

times = pmd.appearedLocalTime.as_matrix()

time = np.array([datetime.strptime(d, '%Y-%m-%dT%H:%M:%S') for d in times])
pmd.head(10)
fig = plt.figure(figsize=(10,5))



m = Basemap(projection='merc',

           llcrnrlat=-60,

           urcrnrlat=65,

           llcrnrlon=-180,

           urcrnrlon=180,

           lat_ts=0,

           resolution='c')



m.drawcoastlines()

m.drawcountries()

m.fillcontinents(color='#888888')

m.drawmapboundary(fill_color='#f4f4f4')

x,y = m(pmd.longitude.tolist(),pmd.latitude.tolist())

m.scatter(x,y,s=3,c="#1292db", lw=0, alpha=1, zorder=5)



plt.show()
original_headers = list(pmd.columns.values)

pmd = pmd._get_numeric_data()

numeric_headers = list(pmd.columns.values)

numpy_array = pmd.as_matrix()

#skl.ensemble.RandomForestClassifier()

pmd.head(1)
pmdtrain = pd.read_csv('../input/300k.csv',nrows=100000,low_memory=False)

pmdtrain = pmdtrain.drop(['closeToWater'],axis=1)

pmdtrain = pmdtrain.drop(pmdtrain.ix[:,'population_density':'cooc_151'].head(0).columns,axis=1)



pmdtrain = pmdtrain._get_numeric_data()





target = [x[1] for x in pmdtrain]

train = [x[2:] for x in pmdtrain]

test = pmd



rf = RandomForestClassifier(n_estimators=100)

rf.fit(train,target)



pmdtrain._get_numeric_data().head(10)