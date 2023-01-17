

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)









import pandas as pd

import numpy as np

from datetime import datetime

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import sklearn as skl

from sklearn.ensemble import RandomForestClassifier


pmd = pd.read_csv('../input/300k.csv',low_memory=False)


len(pmd)



pmd.head()
lats = pmd.latitude.values

lons = pmd.longitude.values

times = pmd.appearedLocalTime.as_matrix()

time = np.array([datetime.strptime(d, '%Y-%m-%dT%H:%M:%S') for d in times])
common_list = [13, 14,15, 16, 17,18, 19, 20, 41, 42]

rare = [ 88, 89, 106, 107, 108,  113, 129, 130, 137, 142]

super_rare = [ 83, 132, 144, 145, 146, 150, 151, 115, 122, 131 ]



pmd['super_class'] = np.nan

pmd.loc[ pmd['class'].isin(common_list),'super_class'] = 1

pmd.loc[ pmd['class'].isin(rare),'super_class'] = 3

pmd.loc[ pmd['class'].isin(super_rare),'super_class'] = 4

pmd.loc[ pd.isnull( pmd['super_class'] ),'super_class'] = 2
pmd['appearedLocalTime']= pd.to_datetime( pmd['appearedLocalTime'] )


fig = plt.figure(figsize=(10,5))



m = Basemap(projection='merc',

           llcrnrlat=-60,

           urcrnrlat=65,

           llcrnrlon=-180,

           urcrnrlon=180,

           lat_ts=0,

           resolution='c')



m.drawcoastlines()

#m.drawcountries()

m.fillcontinents(color='#888888')

m.drawmapboundary(fill_color='#f4f4f4')

import matplotlib.cm as cm



x, y = m(pmd.longitude.tolist(),pmd.latitude.tolist())



c = pmd['class'].tolist()



m.scatter(x,y, s=3, 

                 c=c, 

                 cmap=cm.get_cmap('hot'),

                 lw=0, alpha=1, zorder=5)

plt.title('Pokemon Locations!')

plt.show()
fig = plt.figure(figsize=(10,5))



m = Basemap(projection='merc',

           llcrnrlat=-60,

           urcrnrlat=65,

           llcrnrlon=-180,

           urcrnrlon=180,

           lat_ts=0,

           resolution='c')



m.drawcoastlines()

#m.drawcountries()

m.fillcontinents(color='#888888')

m.drawmapboundary(fill_color='#f4f4f4')

import matplotlib.cm as cm



da = pmd[pmd['super_class'].isin([4])]



x, y = m(da.longitude.tolist(),da.latitude.tolist())



c = da['class'].tolist()





m.scatter(x,y, s=3, 

                 c=c, 

                 cmap=cm.get_cmap('hot'),

                 lw=0, alpha=1, zorder=5)

plt.title('Very Rare Pokemon Locations!')

plt.show()


import datetime as dt

from sklearn.neighbors import KNeighborsClassifier



date_list = pd.date_range(pmd['appearedLocalTime'].min(), pmd['appearedLocalTime'].max())

date_list



res_dict = {}

for num_neigh in np.arange(1,25):

    #print num_neigh

    accuracy_list = []

    

    for train_end_date in date_list[1:-1]:

        test_start_date = train_end_date

        test_end_date = test_start_date + dt.timedelta(days=1)

        #print train_end_date, test_start_date, test_end_date



        train_df = pmd[(pmd['appearedLocalTime']<=train_end_date)]

        test_df = pmd[(pmd['appearedLocalTime']>test_start_date) & (pmd['appearedLocalTime']<=test_end_date)]



        y = train_df['class']

        X = train_df[['latitude', 'longitude']]



        #neigh = KNeighborsClassifier(n_neighbors= num_neigh, weights='uniform')

        neigh = KNeighborsClassifier(n_neighbors= num_neigh, weights='distance')

        neigh.fit(X,y)



        #Z = neigh.predict(X)

        accuracy=neigh.score(test_df[['latitude', 'longitude']], test_df['class'])

        #print accuracy

        accuracy_list.extend([accuracy])

    

    res_dict[num_neigh] = np.nanmean( accuracy_list )

    

print ( res_dict )