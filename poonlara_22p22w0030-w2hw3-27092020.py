# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
pd.options.display.max_rows = 10
pd.options.display.max_columns = 6
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
pd.set_option('precision', 4)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 200)
bnb=pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
bnb.head()
bnb.info()
bnb.isnull().sum()
bnb.shape
bnb['reviews_per_month']=bnb['reviews_per_month'].fillna(0)
bnb.isnull().sum()
bnb.groupby('last_review').size().sort_values(ascending=False)
## Fill Null for Year 1000-01-01
bnb['last_review']=bnb['last_review'].fillna("1900-01-01")
bnb['last_review']=pd.to_datetime(bnb['last_review'],format='%Y-%m-%d')
bnb['last_review_day_tonow']= datetime.now()-bnb['last_review']
bnb['last_review_day_tonow']=bnb['last_review_day_tonow'].dt.days
bnb['last_review_day_tonow'].hist(figsize=(10,5))
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('last_review_day_tonow', fontsize=20)
bnb.groupby('last_review_day_tonow').size().sort_values(ascending=False)
bnb['neighbourhood_group'].hist(figsize=(10,5))
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('neighbourhood_group', fontsize=20)
bnb.groupby('neighbourhood_group').size().sort_values(ascending=False)
bnb.groupby('neighbourhood').size().sort_values(ascending=False)[:100].plot(figsize=(10,5),color='purple', linewidth=5)
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('neighbourhood', fontsize=20)
bnb.groupby('neighbourhood').size().sort_values(ascending=False)
bnb['room_type'].sort_values().hist(figsize=(10,5))
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('room_type', fontsize=20)
bnb.groupby('room_type').size().sort_values(ascending=False)
# Set Price Outlier
a = bnb['price'].quantile(.75)
b = bnb['price'].quantile(0.25)
max = a + (1.5*(a-b))
min = a- (1.5*(a-b))

bnb['price'].hist(bins=50, alpha=0.5,range=[min,max],figsize=(15,10))
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('price', fontsize=20)
bnb.groupby('price').size().sort_values(ascending=False)
plt.figure(figsize=(15,8))
bnb[bnb['room_type']=='Entire home/apt']['price'].hist(bins=50, alpha=0.5,range=[min,max],label="Entire home/apt",color = "skyblue")
bnb[bnb['room_type']=='Private room']['price'].hist(bins=50, alpha=0.5,range=[min,max],label="Private room",color = "orange")
bnb[bnb['room_type']=='Shared room']['price'].hist(bins=50, alpha=0.5,range=[min,max],label="Shared room ",color = "red")
                                                   
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('price', fontsize=20)
plt.legend()
# Set Price Outlier
a = bnb['minimum_nights'].quantile(.75)
b = bnb['minimum_nights'].quantile(0.25)
max = a + (1.5*(a-b))
min = a- (1.5*(a-b))

bnb['minimum_nights'].hist(bins=5, alpha=0.5,range=[min,max],figsize=(15,10))
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('minimum_nights', fontsize=20)
bnb.groupby('minimum_nights').size().sort_values(ascending=False)
# Set Price Outlier
a = bnb['number_of_reviews'].quantile(.75)
b = bnb['number_of_reviews'].quantile(0.25)
max = a + (1.5*(a-b))
min = a- (1.5*(a-b))

bnb['number_of_reviews'].hist(bins=5, alpha=0.5,range=[min,max],figsize=(15,10))
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('number_of_reviews', fontsize=20)
bnb.groupby('number_of_reviews').size().sort_values(ascending=False)
# Set Price Outlier
a = bnb['calculated_host_listings_count'].quantile(.75)
b = bnb['calculated_host_listings_count'].quantile(0.25)
max = a + (1.5*(a-b))
min = a- (1.5*(a-b))

bnb['calculated_host_listings_count'].hist(bins=10, alpha=0.5,range=[min,max],figsize=(15,10))
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('calculated_host_listings_count', fontsize=20)
bnb.groupby('calculated_host_listings_count').size().sort_values(ascending=False)
# Set Price Outlier
a = bnb['availability_365'].quantile(.75)
b = bnb['availability_365'].quantile(0.25)
max = a + (1.5*(a-b))
min = a- (1.5*(a-b))

bnb['availability_365'].hist(bins=5, alpha=0.5,range=[min,max],figsize=(15,10))
plt.xticks(rotation=90,fontsize=15)
plt.xlabel('availability_365', fontsize=20)
bnb.groupby('availability_365').size().sort_values(ascending=False)
bnb.head()
bnb['Year_last_review'] = bnb['last_review'].dt.year
bnb['Month_last_review'] = bnb['last_review'].dt.month
bnb['Day_last_review'] = bnb['last_review'].dt.day
bnb.head()
bnb_new=bnb.drop(columns=['id','name', 'host_id','host_name','neighbourhood_group','neighbourhood','last_review'])
bnb_new
#bnb_new["neighbourhood_group"] = bnb_new["neighbourhood_group"].astype('category')
#bnb_new["neighbourhood"] = bnb_new["neighbourhood"].astype('category')
bnb_new["room_type"] = bnb_new["room_type"].astype('category')
# Create one_hot_encoder function
def one_hot_encoder(dataframe):

  # Select category columns
  cat_df = dataframe.select_dtypes(include=['category']).columns.to_list()

  # Convert to one-hot dataframe
  one_hot_df = pd.get_dummies(dataframe, columns=cat_df, drop_first=True)
  
  return one_hot_df
bnb_df_dummie = one_hot_encoder(bnb_new)
bnb_df_dummie

from sklearn.preprocessing import MinMaxScaler
#MinMaxScaler = MinMaxScaler(feature_range = (0,1))

def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = pd.DataFrame(MinMaxScaler.fit_transform(pd.DataFrame(df[col])),columns=[col])
    return df

#col_to_scale=['latitude','longitude','price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365','last_review_day_tonow','Year_last_review','Month_last_review','Day_last_review']

scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(bnb_df_dummie)
normalized = scaler.transform(bnb_df_dummie)

#bnb_dummie_scale=scaleColumns(bnb_df_dummie,col_to_scale)
from sklearn.cluster import KMeans
w = []
for k in np.arange(2, 16, 1):
    model = KMeans(n_clusters=k, n_jobs=-1).fit(normalized)
    w.append(model.inertia_)
plt.figure(figsize=(15,10))
plt.tick_params(axis='both', labelsize=16)
plt.plot(np.arange(2, 16, 1),w,marker='.',markersize=15)
plt.grid()
cls = KMeans(n_clusters=6, n_jobs=-1)
cls.fit(normalized)
%matplotlib inline
centroid = pd.DataFrame(cls.cluster_centers_, 
                        columns=bnb_df_dummie.columns)

import seaborn as sns
sns.clustermap(centroid, cmap='Oranges')
inversed = scaler.inverse_transform(normalized)
bnb_df_dummie_back=pd.DataFrame(inversed,columns=bnb_df_dummie.columns)
bnb_df_dummie_back
bnb_df_dummie_back['KmeanCluster'] = cls.labels_
bnb_df_dummie_back
bnb_df_dummie_norm=pd.DataFrame(normalized,columns=bnb_df_dummie.columns)
bnb_df_dummie_norm['KmeanCluster'] = cls.labels_
bnb_df_dummie_norm
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

linked = linkage(bnb_df_dummie_norm,'single')
plt.figure(figsize=(20,10))

dendrogram(linked,orientation='top',distance_sort='descending',show_leaf_counts=True)
plt.show()
BBox = (bnb_df_dummie_back.longitude.min(), bnb_df_dummie_back.longitude.max(), 
        bnb_df_dummie_back.latitude.min(), bnb_df_dummie_back.latitude.max())
BBox
fig, ax = plt.subplots()
x = [randint(-10100, -9400)/100 for i in range(30)]
y = [randint(3700, 4000)/100 for i in range(30)]
ax.plot(x, y, 'bo')
mplleaflet.display(fig=fig)

#ruh_m = plt.imread('https://raw.githubusercontent.com/imetanon/botnoi-dsessential/master/boston-map.png')
#import mplleaflet
fig, map = plt.subplots(figsize = (20,19))
map.scatter(bnb_df_dummie_back.longitude, bnb_df_dummie_back.latitude, zorder=1, alpha= 0.3, c=bnb_df_dummie_back.KmeanCluster, s=15)
map.set_title('Plotting Airbnb Listed Places on NYC Map')
map.set_xlim(BBox[0],BBox[1])
map.set_ylim(BBox[2],BBox[3])

#mplleaflet.display(fig=fig)
#map.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')