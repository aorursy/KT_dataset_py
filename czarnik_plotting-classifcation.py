

import pandas as pd
import numpy as np
import zipfile
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as pl
from mpl_toolkits.basemap import Basemap
import seaborn as sns


#import train and test
z = zipfile.ZipFile('train.csv.zip')
z2= zipfile.ZipFile('test.csv.zip')

train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])
test = pd.read_csv(z2.open('test.csv'), parse_dates=['Dates'])

train['Year'] = train['Dates'].map(lambda x: x.year)
train['Week'] = train['Dates'].map(lambda x: x.week)
train['Hour'] = train['Dates'].map(lambda x: x.hour)
#making a testing set
test['Year'] = test['Dates'].map(lambda x: x.year)
test['Week'] = test['Dates'].map(lambda x: x.week)
test['Hour'] = test['Dates'].map(lambda x: x.hour)

x_train = train[['X', 'Y','Year','Week','Hour']] 
y = train['Category'].astype('category')
    
n = len(set(train['Category']))

##find the number of categories^^

x_test = test[['X', 'Y','Year','Week','Hour']]
##find the features you want to test
knn = KNeighborsClassifier(n_neighbors=10)
    
knn.fit(x_train, y)
predicted = knn.predict(x_test)
    
pl.figure(figsize=(20,5))
map_extent = [-122.53, 37.68, -122.35, 37.83]

m = Basemap(llcrnrlon=map_extent[0], llcrnrlat=map_extent[1],
             urcrnrlon=map_extent[2], urcrnrlat=map_extent[3],projection='lcc',resolution='f', epsg=4269)
m.drawcoastlines(color='red')
m.drawcountries()
m.fillcontinents(color = 'gray')

longitudes = np.arange(-122.53, -122.35, .02)
latitudes = np.arange(37.68, 37.83, .02)

m.drawparallels(latitudes,labels=[1,1,0,0])
m.drawmeridians(longitudes,labels=[0,0,0,1])
    
asp = 1

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

#Get rid of the bad lat/longs
test = test.dropna()
test['predicted'] = predicted
test_cat = test[test.predicted== 'ASSAULT'] 
test_cat = test_cat #Can't use all the data and complete within 600 sec :(

ax = sns.kdeplot(test_cat.X, test_cat.Y,n_levels=10, clip=clipsize, aspect=1/asp)
ax= pl.gca()
    
pl.show()

##testing values
msk = np.random.rand(len(train)) < 0.7
knn_train = train[msk]
knn_test = train[~msk]

n = len(knn_test)

x = knn_test[['X', 'Y','Year','Week','Hour']]

y = knn_test['Category'].astype('category')
y_pred = knn.predict(x)
actual = knn_test['Category'].astype('category')

##trying to enumearte things to check
enumarete_array = np.array(list(enumerate(set(train['Category']))))[:,0]

print(knn.score(x,actual, sample_weight=None))

from sklearn import metrics
print(metrics.adjusted_rand_score(actual, y_pred))

