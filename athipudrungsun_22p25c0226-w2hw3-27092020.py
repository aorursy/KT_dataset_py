# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import urllib

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv').set_index('id')

df.head(10)
df.columns
df.value_counts('neighbourhood_group').index
air = df[['host_id', 'neighbourhood_group', 'neighbourhood',

       'latitude', 'longitude', 'room_type', 'price', 'minimum_nights',

       'number_of_reviews', 'reviews_per_month',

       'calculated_host_listings_count', 'availability_365']]
air['reviews_per_month'].fillna(0, inplace = True)
air['reviews_per_month'].isna().sum()
air.isna().sum()
price_review = air[['neighbourhood_group','price','reviews_per_month', 'calculated_host_listings_count']]

# remove price outlier

price_review = price_review[(price_review.price < 6000)&(price_review.reviews_per_month < 30)]
plt.figure(figsize=(10,4))

sns.pairplot(price_review,hue='neighbourhood_group', height=3, aspect=2)
from sklearn import preprocessing

from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch # draw dendrogram



air_s = air[['price', 'minimum_nights', 'reviews_per_month','calculated_host_listings_count']]



# dummy_n_group = pd.get_dummies(air['neighbourhood_group'], prefix='neighbourhood_group')

# dummy_room_type = pd.get_dummies(air['room_type'], prefix='room_type')



# air_hi = pd.concat([air_s, dummy_n_group, dummy_room_type], axis='columns')

cols = air_s.columns



pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True) # support only positive value

scale_air = pt.fit_transform(air_s)
print(cols)
scale_air
air_scaled = pd.DataFrame(scale_air, columns=cols)

air_scaled.head(3)
# run ไม่ผ่าน over memory

# fig, ax=plt.subplots(figsize=(20, 7))

# dg=sch.dendrogram(sch.linkage(air_scaled, method='ward'), ax=ax, labels=df['id'].values)
#classification along with neighbourhood_group

location = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']

suv_cols = ['price', 'minimum_nights', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

air_sub = air[suv_cols]

df_manhuttan = air_sub[air.neighbourhood_group == 'Manhattan']

df_manhuttan.head(5)
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True) # support only positive value

s_man = pt.fit_transform(df_manhuttan)

man_scaled = pd.DataFrame(s_man, columns=suv_cols)

man_scaled.head(3)
#create dendogram 

fig, ax=plt.subplots(figsize=(20, 7))

dg=sch.dendrogram(sch.linkage(man_scaled, method='ward'), ax=ax, labels=df_manhuttan.index)
sns.clustermap(man_scaled, col_cluster=False, cmap="Greens", figsize=(10, 30));
hc=AgglomerativeClustering(n_clusters=3, linkage='ward')

hc
hc.fit(man_scaled)
hc.labels_
df_manhuttan['cluster']=hc.labels_

df_manhuttan.head()
df_manhuttan.groupby('cluster').agg(['mean', 'std'])

# df_manhuttan.merge(df[['latitude', 'longitude']], on=df_manhuttan.index)

df_manhuttan = pd.merge(df_manhuttan, df[['latitude', 'longitude']], left_on=df_manhuttan.index, right_index=True)

df_manhuttan.info()
df_manhuttan = df_manhuttan.drop('key_0', axis='columns')

df_manhuttan.info()
#The follow code thanks to : dgomonov  (link:https://www.kaggle.com/dgomonov/data-exploration-on-nyc-airbnb)

#initializing the figure size

plt.figure(figsize=(20,8))



c_1=df_manhuttan

# c_2 = df_manhuttan[df_manhuttan.cluster == 1]

# c_3 = df_manhuttan[df_manhuttan.cluster == 2]



img = mpimg.imread('/kaggle/input/new-york-city-airbnb-open-data/New_York_City_.png', 0)



#scaling the image based on the latitude and longitude max and mins for proper output

# plt.imshow(img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])

plt.imshow(img,zorder=0,extent=[-74.05, -73.9, 40.7,40.92])

ax=plt.gca()



# #using scatterplot again

c_1.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='cluster', ax=ax, 

           cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.2, zorder=3)



plt.legend()

plt.show()
