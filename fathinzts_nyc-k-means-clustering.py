!pip install kaggle

from google.colab import files

files.upload()

import pandas as pd
!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d new-york-city/nyc-property-sales
!unzip nyc-property-sales.zip
data = pd.read_csv('nyc-rolling-sales.csv')
data.head()
ds = data[['TOTAL UNITS','BLOCK','SALE PRICE','YEAR BUILT','LOT','ZIP CODE']] #kolom data yang akan digunakan
ds.head()
#check data kosong

ds.isna().sum()
ds['YEAR BUILT'].describe()

#len(ds) #cek brp data yang ada
ds2 = ds[ds['YEAR BUILT'] > 1965 ]
len(ds),len(ds2)
ds2.head()
from sklearn import preprocessing
#convert all value to decimal number

minmax = preprocessing.MinMaxScaler().fit_transform(ds2.drop('SALE PRICE',axis=1))
minmax
#convert numpy to pandas

ds3 = pd.DataFrame(minmax, index=ds2.index, columns=ds2.columns[:-1])
ds3
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
#how to know cluster?

scr=[]

for i in range(1,20):

  score=KMeans(n_clusters=i).fit(ds3).score(ds3)

  print(score)

  scr.append(score)
#see distribution efficient or not with mathplotlib

plt.plot(scr)
#from diagram we know 2.5 is the best cluster

km = KMeans(n_clusters=3)

#kenapa dipke 3? krn sblmnya nyoba 2.5 dibilang float can't be convert to integer. maunya yg bilangan pas aja si doi :"

#jadi dipke yang paling deket dengan 2.5

km.fit(ds3)
km.labels_
ds3['cluster'] = km.labels_ #adding column cluster
ds3
#see cluster in histogram

plt.hist(ds3['cluster'])
import seaborn as sns

sns.pairplot(ds3,hue='cluster')
ds3
ds3['SALE PRICE'] = ds2['SALE PRICE']

ds3