%matplotlib inline

from scipy.cluster.hierarchy import dendrogram, linkage

from matplotlib import pyplot as plt

import pandas as pd
df_origin = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df = df_origin.copy()
drop_elements = ['last_review', 'host_name','id']

df.drop(drop_elements, axis = 1, inplace= True)

df.fillna({'reviews_per_month':0}, inplace=True)
df.head(10)
df.info()
n_row, n_col = df.shape



print("Number of row: ",n_row)

print("Number of column: ",n_col)
import matplotlib.pyplot as plt

import seaborn as sns



fig = plt.subplots(figsize = (12,5))

sns.countplot(x = 'room_type', hue = 'neighbourhood_group', data = df)
df['neighbourhood_group'].value_counts()
df.price.quantile(.99)
plt.figure(figsize = (16,12))

temp = df[df.price < 799.0]

sns.violinplot(x = 'neighbourhood_group', y = 'price', data = temp)
df.number_of_reviews.quantile(.99)
plt.figure(figsize = (16,12))

temp = df[df.number_of_reviews < 214.0]

sns.violinplot(x = 'neighbourhood_group', y = 'number_of_reviews', data = temp)
df.minimum_nights.quantile(.99)
plt.figure(figsize = (16,12))

temp = df[df.minimum_nights < 45]

sns.violinplot(x = 'neighbourhood_group', y = 'minimum_nights', data = temp)
def show_result(Z, title, xlabel, ylabel):

    fig = plt.figure(figsize=(60, 50))

    dendrogram(Z, labels=df_grouped.neighbourhood.to_list(), leaf_rotation=0, orientation="left")

    my_palette = plt.cm.get_cmap("Accent", 3)

    plt.title(title,fontsize=50)

    plt.xlabel(xlabel,fontsize=30)

    plt.ylabel(ylabel,fontsize=30)

    plt.show()
df_grouped = df.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','price']]

list_grouped_price_mean = [ [price] for price in df_grouped.price.to_list()]

df_grouped.head(10)
Z = linkage(list_grouped_price_mean, 'complete')
title = 'Clustering neighbourhood in NYC by Average price of each neighbourhood'

xlabel = 'Average Price'

ylabel = 'Neighbourhood'



show_result(Z, title, xlabel, ylabel)
df_grouped = df.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','number_of_reviews']]

list_grouped_reviewNo_mean = [ [reviewNo] for reviewNo in df_grouped.number_of_reviews.to_list()]

df_grouped.head(10)
Z = linkage(list_grouped_reviewNo_mean, 'complete')
title = 'Clustering neighbourhood in NYC by Average number of reviews of each neighbourhood'

xlabel = 'Average number of reviews'

ylabel = 'Neighbourhood'



show_result(Z, title, xlabel, ylabel)
df_grouped = df.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','minimum_nights']]

list_grouped_night_mean = [ [night] for night in df_grouped.minimum_nights.to_list()]

df_grouped.head(10)
Z = linkage(list_grouped_night_mean, 'complete')
title = 'Clustering neighbourhood in NYC by Average minimum nights of each neighbourhood'

xlabel = 'Average minimum nights'

ylabel = 'Neighbourhood'



show_result(Z, title, xlabel, ylabel)
df_grouped = df.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','number_of_reviews','price']]

list_grouped_reviewAndprice_mean = [ [review,price] for review, price in zip(df_grouped.number_of_reviews.to_list(),df_grouped.price.to_list())]

df_grouped.head(10)
from sklearn.preprocessing import MinMaxScaler
# Scale the data for clustering easier

scaler = MinMaxScaler()

scaler.fit(list_grouped_reviewAndprice_mean)

list_grouped_reviewAndprice_mean_scale = scaler.transform(list_grouped_reviewAndprice_mean)
Z = linkage(list_grouped_reviewAndprice_mean_scale, 'complete')
title = ' Clustering neighbourhood in NYC by Average price of each neighbourhood and Average number of reviews of each neighbourhood'

xlabel = 'Average price and Average number of reviews'

ylabel = 'Neighbourhood'



show_result(Z, title, xlabel, ylabel)
df_grouped = df.groupby('neighbourhood').mean().reset_index().loc[:,['neighbourhood','number_of_reviews','price', 'minimum_nights']]

list_grouped_reviewAndpriceAndnight_mean = [ [review,price,night] for review, price, night in zip(df_grouped.number_of_reviews.to_list()

                                                                                                  ,df_grouped.price.to_list()

                                                                                                  ,df_grouped.minimum_nights.to_list())]

df_grouped.head(10)
# Scale the data for clustering easier

scaler = MinMaxScaler()

scaler.fit(list_grouped_reviewAndpriceAndnight_mean)

list_grouped_reviewAndpriceAndnight_mean_scale = scaler.transform(list_grouped_reviewAndpriceAndnight_mean)
Z = linkage(list_grouped_reviewAndpriceAndnight_mean_scale, 'complete')
title = 'Clustering neighbourhood in NYC by Average price of each neighbourhood and Average number of reviews of each neighbourhood and Average minimum nights of each neighbourhood'

xlabel = 'Average price and Average number of reviews and Average minimum nights'

ylabel = 'Neighbourhood'



show_result(Z, title, xlabel, ylabel)