# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_airbnb=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
print("Row : "+str(data_airbnb.shape[0]))
print("Column : "+str(data_airbnb.shape[1]))
#show feature name & dtype
data_airbnb.dtypes
data_airbnb.head()
data_airbnb.isnull().sum()
data_airbnb.info()
data_airbnb['name'].value_counts()
data_airbnb['host_name'].value_counts()
data_airbnb['neighbourhood_group'].value_counts()
data_airbnb['neighbourhood'].value_counts()
data_airbnb['room_type'].value_counts()
data_airbnb['last_review'].value_counts()
data_airbnb['last_review'][0:5]
data_airbnb["id"].hist()
data_airbnb["host_id"].hist()
data_airbnb["price"].hist()
ax = sns.boxplot(x=data_airbnb["price"])
data_airbnb["minimum_nights"].value_counts()
ax = sns.boxplot(x=data_airbnb["minimum_nights"])
ax = sns.boxplot(x=data_airbnb["calculated_host_listings_count"])
ax = sns.boxplot(x=data_airbnb["availability_365"])
data_airbnb.duplicated().sum()
data_airbnb.drop_duplicates(inplace=True)
data_airbnb.info()
data_airbnb.drop(['name','id','host_name','last_review'], axis=1, inplace=True)
data_airbnb.shape[1]

data_airbnb = data_airbnb.dropna()
data_airbnb.shape
plt.figure(figsize=(20,10))
sns.scatterplot(data_airbnb["longitude"],data_airbnb["latitude"],hue=data_airbnb["neighbourhood_group"])
plt.ioff()
plt.figure(figsize=(20,10))
sns.scatterplot(data_airbnb["longitude"],data_airbnb["latitude"],hue=data_airbnb["room_type"])
plt.ioff()
plt.figure(figsize=(20,10))
sns.scatterplot(data_airbnb["longitude"],data_airbnb["latitude"],hue=data_airbnb["calculated_host_listings_count"])
plt.ioff()
plt.figure(figsize=(20,10))
sns.scatterplot(data_airbnb["longitude"],data_airbnb["latitude"],hue=data_airbnb["availability_365"])
plt.ioff()
corr = data_airbnb.corr(method="pearson")
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)
data_airbnb.columns
data_airbnb.shape
data_airbnb["neighbourhood_group"].value_counts()
data_airbnb.groupby("neighbourhood_group").mean()
data_fil_neighbourhood_group = data_airbnb.groupby("neighbourhood_group").mean().reset_index().loc[:,['neighbourhood_group','price']]
data_fil_neighbourhood_group.head()
list_data_fil_neighbourhood_group = [ [price] for price in data_fil_neighbourhood_group["price"].to_list()]
list_data_fil_neighbourhood_group
Z = linkage(list_data_fil_neighbourhood_group, 'complete')
fig = plt.figure(figsize=(20, 10))
dendrogram(Z, labels=data_fil_neighbourhood_group["neighbourhood_group"].to_list())
plt.ylabel('Price')
plt.xlabel('Neighbourhood_group')
plt.show()
data_airbnb["neighbourhood"].value_counts()
data_airbnb["neighbourhood"].value_counts()
data_fil_neighbourhood = data_airbnb.groupby("neighbourhood").mean().reset_index().loc[:,['neighbourhood','price']]
data_fil_neighbourhood.head()
list_data_fil_neighbourhood = [ [price] for price in data_fil_neighbourhood["price"].to_list()]
list_data_fil_neighbourhood
Z = linkage(list_data_fil_neighbourhood, 'complete')
fig = plt.figure(figsize=(20, 10))
dendrogram(Z, labels=data_fil_neighbourhood["neighbourhood"].to_list())
plt.ylabel('Price')
plt.xlabel('Neighbourhood')
plt.show()
data_fil_room_type = data_airbnb.groupby("room_type").mean().reset_index().loc[:,['price','room_type']]
data_fil_room_type.head()
list_data_fil_room_type = [ [price] for price in data_fil_room_type["price"].to_list()]
list_data_fil_room_type
Z = linkage(list_data_fil_room_type, 'complete')

fig = plt.figure(figsize=(20, 10))
dendrogram(Z, labels=data_fil_room_type["room_type"].to_list())
plt.ylabel('Price')
plt.xlabel('room_type')
plt.show()