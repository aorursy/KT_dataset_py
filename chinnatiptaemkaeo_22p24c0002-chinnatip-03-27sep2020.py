# Required lib
%matplotlib inline
from scipy.cluster.hierarchy import dendrogram, linkage
# import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import os
# Show import dataset
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load ABNB NY Dataset from 'kaggle'
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# Fill NAN
df.fillna({'reviews_per_month':0 , 'last_review': 0}, inplace=True)

# Drop Un-used column
drop_elements = ['last_review', 'host_name','id']
df.drop(drop_elements, axis = 1, inplace= True)

# Show
df.head(100)
# Show DF DataType
df.info()
# Show DF size
[row , col] = df.shape
print(f"Record have {row} rows | {col} columns")
df_grouped       = df.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','price']]
neibourhood_mean = [ [price] for price in df_grouped['price'].to_list()]
df_grouped.head()
# Hierachy Clutering
Z = linkage(neibourhood_mean, method='ward')
# Plot Dendrogram
fig = plt.figure(figsize=(100, 40)) # fix size 100 x 40 pixel
dendrogram(Z, labels=df_grouped['neighbourhood'].to_list(), leaf_rotation=0, orientation="left")
my_palette = plt.cm.get_cmap("Accent", 1)
plt.show()
# Show Neighbour group
df['neighbourhood_group'].value_counts()
# show ROOM and Neighbour group data ..
fig = plt.subplots(figsize = (12,5))
sns.countplot(x = 'room_type', hue = 'neighbourhood_group', data = df)