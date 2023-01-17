import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc

from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()
print(data.shape)
data.info()
room_type = {'Private room':0, 'Entire home/apt':1, 'Shared room':2}
Room = []

for index,row in data.iterrows():
    Room.append(room_type[row['room_type']])
    
data.drop(['id','room_type','host_name','neighbourhood_group','neighbourhood','name','last_review','reviews_per_month','calculated_host_listings_count','availability_365','minimum_nights','number_of_reviews'],axis=1,inplace=True)
data['room_type'] = Room

data['latitude'] = data['latitude'].round(1)
data['longitude'] = data['longitude'].round(1)
data.info()
data.head(10)
remove_n = round(data.shape[0] * 0.5) # drop about 50%

drop_indices = np.random.choice(data.index, remove_n, replace=False)
data_subset = data.drop(drop_indices)
print(data_subset.shape)
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(data_subset, method='ward'))