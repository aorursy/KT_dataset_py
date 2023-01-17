import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from operator import itemgetter

# Input data files are available in the "../input/" directory.

import os
input_folder = "../input/competitive-data-science-predict-future-sales"
file_names = (os.listdir(input_folder))

shops = pd.read_csv(os.path.join(input_folder,'shops.csv'))
shops['city']=shops.shop_name.apply(str.split).apply(itemgetter(0)).apply(str.strip,args='!')
shops.head()
shops[['lat','long']]=pd.read_csv('../input/geo-info/coordinates.csv',index_col=0,dtype=np.float)
shops['city'] = pd.read_csv('../input/geo-info/city_name_eng.csv',index_col=0,header=None)

shops
shops_summary=shops[~shops.lat.isna()].groupby('city').agg({'lat':np.mean,'long':np.mean,'shop_id':len}).reset_index()
shops_summary.rename({'shop_id':'city_cnt'},inplace=True,axis=1)
shops_summary
%matplotlib inline
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import builtins

plt.figure(figsize=(14,10))

map = Basemap(llcrnrlon=19.5,llcrnrlat=35,urcrnrlon=140,urcrnrlat=75)

map.drawcoastlines()
map.drawcountries()

x, y = map(shops_summary.long, shops_summary.lat)

map.scatter(x, y, marker='D',color='m')
for i in range(len(x)):
    plt.text(x[i],y[i],shops_summary.loc[i,'city'])


plt.show()
