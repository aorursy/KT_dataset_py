# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.cluster import DBSCAN
# Importing the dataset

dataset = pd.read_csv('/kaggle/input/weather-stations.csv')
dataset.head()
dataset.shape
dataset.nunique()
dataset.isna().sum()

dataset.dropna(subset=['Tm', 'Tx', 'Tn'], inplace=True)

print ("After Dropping Rows that contains NaN on Mean, Max, Min Temperature Column: ", dataset.shape)
from mpl_toolkits.basemap import Basemap

import matplotlib

from PIL import Image

import matplotlib.pyplot as plt

#print (matplotlib.__version__)

from pylab import rcParams

%matplotlib inline

rcParams['figure.figsize'] = (14,10)





llon=-140

ulon=-50

llat=40

ulat=75



# selecting the boundaries of the map from lattitude and longitude 



dataset = dataset[(dataset['Long'] > llon) & (dataset['Long'] < ulon) & (dataset['Lat'] > llat) &(dataset['Lat'] < ulat)]







my_map = Basemap(projection='merc',

            resolution = 'l', area_thresh = 1000.0,

            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)

            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)



my_map.drawcoastlines()

my_map.drawcountries()

my_map.drawlsmask(land_color='orange', ocean_color='skyblue')

#my_map.shadedrelief()

my_map.bluemarble()

# To collect data based on stations        



xs,ys = my_map(np.asarray(dataset.Long), np.asarray(dataset.Lat))

dataset['xm']= xs.tolist()

dataset['ym'] =ys.tolist()



#Visualization1

for index,row in dataset.iterrows():

#   x,y = my_map(row.Long, row.Lat)

   my_map.plot(row.xm, row.ym,markerfacecolor ='lime',markeredgecolor='pink', marker='s', markersize= 10, alpha = 0.4)

#plt.text(x,y,stn)

plt.title("Weather Stations in Canada", fontsize=14)

plt.savefig("Canada_WS.png", dpi=300)

plt.show()
from sklearn.cluster import DBSCAN

import sklearn.utils

from sklearn.preprocessing import StandardScaler

dataset_clus_temp = dataset[["Tm", "Tx", "Tn", "xm", "ym"]]

dataset_clus_temp = StandardScaler().fit_transform(dataset_clus_temp)



db = DBSCAN(eps=0.3, min_samples=10).fit(dataset_clus_temp)

labels = db.labels_

print (labels[500:560])

dataset["Clus_Db"]=labels



realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)

clusterNum = len(set(labels))
set(labels)
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

from pylab import rcParams

%matplotlib inline

rcParams['figure.figsize'] = (14,10)



my_map = Basemap(projection='merc',

            resolution = 'l', area_thresh = 1000.0,

            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)

            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)



my_map.drawcoastlines()

my_map.drawcountries()

my_map.drawlsmask(land_color='orange', ocean_color='skyblue')

my_map.etopo()



# To create a color map

colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))







#Visualization1

for clust_number in set(labels):

    c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])

    clust_set = dataset[dataset.Clus_Db == clust_number]                    

    my_map.scatter(clust_set.xm, clust_set.ym, color =c,  marker='o', s= 40, alpha = 0.65)

    if clust_number != -1:

        cenx=np.mean(clust_set.xm) 

        ceny=np.mean(clust_set.ym) 

        plt.text(cenx,ceny,str(clust_number), fontsize=30, color='red',)

        print ("Cluster "+str(clust_number)+', Average Mean Temp: '+ str(np.mean(clust_set.Tm)))

plt.title(r"Weather Stations in Canada Clustered (1): $ \epsilon = 0.3$", fontsize=14)        

plt.savefig("etopo_cluster.png", dpi=300)
dataset_copy = dataset.copy()

dataset_clus_temp_P = dataset_copy[["Tm", "Tx", "Tn", "xm", "ym", "P"]]



dataset_clus_temp_P.dropna(subset=["Tm", "Tx", "Tn", "xm", "ym", "P"], inplace=True)

print ("After Dropping Rows that contains NaN on Precipitation Column: ", dataset_clus_temp_P.shape)



print (dataset_clus_temp_P.head(6))


#print (weather_df_clus_temp.shape)

dataset_clus_temp_P_arr = dataset_clus_temp_P[["Tm", "Tx", "Tn", "xm", "ym", "P"]]

dataset_clus_temp_P_arr = StandardScaler().fit_transform(dataset_clus_temp_P_arr)



db_P = DBSCAN(eps=0.5, min_samples=10).fit(dataset_clus_temp_P_arr)

# # create an array of zeroes of same size as db.labels_. db.labels_ is an array containing labels for 

labels_P = db_P.labels_

print(labels_P[500:560])

print (labels_P.dtype)

#print(np.isnan(labels_P).any())

dataset_clus_temp_P["Clus_Db_"]=labels_P
realClusterNum_P=len(set(labels_P)) - (1 if -1 in labels_P else 0)

clusterNum_P = len(set(labels_P)) 





print (set(labels_P))
rcParams['figure.figsize'] = (14,10)



my_map1 = Basemap(projection='merc',

            resolution = 'l', area_thresh = 1000.0,

            llcrnrlon=llon, llcrnrlat=llat, #min longitude (llcrnrlon) and latitude (llcrnrlat)

            urcrnrlon=ulon, urcrnrlat=ulat) #max longitude (urcrnrlon) and latitude (urcrnrlat)



my_map1.drawcoastlines()

my_map1.drawcountries()

my_map1.drawlsmask(land_color='orange', ocean_color='skyblue')

my_map1.etopo()



# To create a color map

colors1 = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum_P))







#Visualization1

for clust_number_P in set(labels_P):

    c=(([0.4,0.4,0.4]) if clust_number_P == -1 else colors1[np.int(clust_number_P)])

    clust_set_P = dataset_clus_temp_P[dataset_clus_temp_P.Clus_Db_ == clust_number_P]                    

    my_map.scatter(clust_set_P.xm, clust_set_P.ym, color =c,  marker='o', s= 40, alpha = 0.65)

    if clust_number_P != -1:

        cenx=np.mean(clust_set_P.xm) 

        ceny=np.mean(clust_set_P.ym) 

        plt.text(cenx,ceny,str(clust_number_P), fontsize=30, color='red',)

        print ("Cluster "+str(clust_number_P)+', Average Mean Temp: '+ str(np.mean(clust_set_P.Tm)))

        print ("Cluster "+str(clust_number_P)+', Average Mean Precipitation: '+ str(np.mean(clust_set_P.P)))

plt.savefig("etopo_cluster_preci.png", dpi=300)