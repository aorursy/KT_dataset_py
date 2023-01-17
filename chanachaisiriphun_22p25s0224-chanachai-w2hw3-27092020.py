from matplotlib import pyplot as plt
import pandas as pd
airbnb_og = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb = airbnb_og.copy()
airbnb.head(5)
airbnb.info()
airbnb.dtypes
airbnb.drop(['id','host_name','last_review'], axis=1, inplace=True)
airbnb.head(5)
airbnb.fillna({'reviews_per_month':0}, inplace=True)
airbnb.reviews_per_month.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.subplots(figsize = (5,3))
sns.countplot(x = 'neighbourhood_group', hue = 'room_type', data = airbnb)
airbnb['neighbourhood_group'].value_counts()
sb = airbnb[airbnb.price < 500]
sns.violinplot(x = 'neighbourhood_group', y = 'price', data = sb)
from scipy.cluster.hierarchy import dendrogram, linkage
def dendrogram_result(Z, title, xlabel, ylabel):
    fig = plt.figure(figsize=(20, 100))
    dendrogram(Z, labels=airbnb_group.neighbourhood.to_list(), leaf_rotation=0, orientation="left")
    plt.title(title,fontsize=30)
    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(ylabel,fontsize=30)
airbnb_group = airbnb.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','price']]
group_price_list_mean = [[price] for price in airbnb_group.price.to_list()]
airbnb_group.head(5)
Z = linkage(group_price_list_mean, 'complete')
title = 'Clustering in airbnb NYC by average price of each neighbourhood'
xlabel = 'Average Price'
ylabel = 'Neighbourhood'

dendrogram_result(Z, title, xlabel, ylabel)