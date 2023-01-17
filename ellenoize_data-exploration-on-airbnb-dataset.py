import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium as fl
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
# Reading data
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
# General data exploration

print('Number of examples: {}\n'.format(len(data)))
print('Columns:\n{}\n'.format(data.columns))
print('Types of data:\n{}\n'.format(data.dtypes))
print('Number of NaN values:\n{}'.format(data.isna().sum()))
display(data.head())
# Let's have a closer look at some values

fig, axs = plt.subplots(2, figsize=(15, 10))

# Plotting distributions
sns.distplot(data.price, ax=axs[0]).set_title('Price Distribution')
sns.distplot(data.number_of_reviews, ax=axs[1]).set_title('N of Reviews Distribution')

print('Min price: {}      Avg price: {}      Max price: {}'.format(data.price.min(),
                                                                   data.price.mean(),
                                                                   data.price.max()))

print('Min n reviews: {}      Avg n reviews: {}      Max n reviews: {}'.format(data.number_of_reviews.min(),
                                                                       data.number_of_reviews.mean(),
                                                                       data.number_of_reviews.max()))
# Deleting hosts with 0 price
data = data[data.price != 0]

# Plotting map of NYC
MAP = fl.Map([data.latitude.mean(), data.longitude.mean()])

# Plotting on map hosts with the most expensive and cheapest prices
indxs_price_high = data.price.nlargest(30).index.values
indxs_price_low = data.price.nsmallest(30).index.values

for i in indxs_price_high:
    
    pos = [data.latitude.loc[i], data.longitude.loc[i]]
    message = "Price: {}".format(data.price[i])
    icon = fl.Icon(color='blue', icon='star-o')
    fl.Marker(pos, popup=message, icon=icon).add_to(MAP)
    
for i in indxs_price_low:
    
    pos = [data.latitude.loc[i], data.longitude.loc[i]]
    message = "Price: {}".format(data.price[i])
    icon = fl.Icon(color='red', icon='star-o')
    fl.Marker(pos, popup=message, icon=icon).add_to(MAP)
    
    
MAP
neighbour_data = data.groupby('neighbourhood_group')['price']
neighbour_data_mean = neighbour_data.mean().sort_values(ascending=False)

plt.figure(figsize=(13, 10))
bar_plot = sns.barplot(neighbour_data_mean.index.values, neighbour_data_mean)
plt.title('Mean prices in each neighbour group')

mean_indexs = neighbour_data_mean.index.values
# Adding prices to the bar plots
for i in range(len(mean_indexs)):
    nbhood = mean_indexs[i]
    bar_plot.text(i, neighbour_data_mean.get(nbhood), round(neighbour_data_mean.get(i), 2), color='black', ha="center")

neighbour_data_median = neighbour_data.median().sort_values(ascending=False)

plt.figure(figsize=(13, 10))
bar_plot = sns.barplot(neighbour_data_median.index.values, neighbour_data_median)
plt.title('Median prices in each neighbour group')

median_indexs = neighbour_data_median.index.values
for i in range(len(median_indexs)):
    nbhood = median_indexs[i]
    bar_plot.text(i, neighbour_data_median.get(nbhood), round(neighbour_data_median.get(i), 2), color='black', ha="center")

plt.show()
roomtype_data = data.groupby('room_type')['price']
roomtype_data_mean = roomtype_data.mean().sort_values(ascending=False)

plt.figure(figsize=(13, 10))
bar_plot = sns.barplot(roomtype_data_mean.index.values, roomtype_data_mean)
plt.title('Mean prices in each neighbour group')

mean_indexs = roomtype_data_mean.index.values

# Adding prices to the bar plots
for i in range(len(mean_indexs)):
    nbhood = mean_indexs[i]
    bar_plot.text(i, roomtype_data_mean.get(nbhood), round(roomtype_data_mean.get(i), 2), color='black', ha="center")

roomtype_data_median = roomtype_data.median().sort_values(ascending=False)

plt.figure(figsize=(13, 10))
bar_plot = sns.barplot(roomtype_data_median.index.values, roomtype_data_median)
plt.title('Median prices in each neighbour group')

median_indexs = roomtype_data_median.index.values
for i in range(len(median_indexs)):
    nbhood = median_indexs[i]
    bar_plot.text(i, roomtype_data_median.get(nbhood), round(roomtype_data_median.get(i), 2), color='black', ha="center")

plt.show()
sns.set()
sns.set_style('darkgrid')

fig, axs = plt.subplots(3, figsize=(17,13))

axs[0].scatter(data.number_of_reviews, data.price)
axs[0].set(xlabel = 'Number of reviews', ylabel = 'Price')

axs[1].scatter(data.reviews_per_month, data.price, c='red')
axs[1].set(xlabel = 'Reviews per month', ylabel = 'Price')

axs[2].scatter(data.availability_365, data.price, c='green')
axs[2].set(xlabel = 'Number of days host is available', ylabel = 'Price')

plt.show()
model = CatBoostRegressor(iterations=600, custom_metric=['RMSE'], ) 

# Selecting features to train the model
features = ['neighbourhood', 'room_type', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'latitude', 'longitude']
X_train, X_eval, y_train, y_eval = train_test_split(data[features], data.price, train_size=0.8, random_state=42)

train_Pool = Pool(X_train, y_train, cat_features=[0,1])
eval_Pool = Pool(X_eval, y_eval, cat_features=[0,1])

model.fit(train_Pool, verbose=False, eval_set=eval_Pool, plot=True)
model.get_best_score()
