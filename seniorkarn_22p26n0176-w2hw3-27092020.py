import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df
df.info()
df.isnull().sum()
df = df.drop(['id','host_id','host_name', 'latitude', 'longitude','last_review'],axis=1)
df
df.fillna({'reviews_per_month':0}, inplace=True) #float64
df.fillna({'name':"none"}, inplace=True) #object
df.isnull().sum()
df['neighbourhood'].value_counts()
df['neighbourhood'].unique()
nh_mean = df.groupby(["neighbourhood"], as_index=False).mean()
nh_mean

nh_nr = nh_mean[['neighbourhood','number_of_reviews']]

nh_nr
lstreview_mean = []
for lstreview in nh_nr.number_of_reviews.to_list():
    lstreview_mean.append([lstreview])

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

Z = hierarchy.linkage(lstreview_mean, 'complete')
plt.figure()
dn = hierarchy.dendrogram(Z)
