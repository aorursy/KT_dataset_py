import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
Airbnb_data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

print(Airbnb_data.info())

Airbnb_data.sample(5)
print(Airbnb_data.isnull().sum())

Airbnb_data.describe()
temp_data = Airbnb_data[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(Airbnb_data.corr(), ax=ax, annot=True, linewidth=0.05, fmt='.2f', cmap='magma')
plt.figure(figsize=(15, 10))

sns.scatterplot(x="longitude", y="latitude", hue='room_type',  size='price', data=Airbnb_data,  sizes=(20, 300), palette="Set2")
room_types = Airbnb_data.drop_duplicates(subset='room_type', keep='first' , inplace=False)['room_type']

room_types
sns.boxplot(x="room_type", y="price", data=Airbnb_data)
neighbourhood_group = Airbnb_data.drop_duplicates(subset='neighbourhood_group', keep='first' , inplace=False)['neighbourhood_group']

print(neighbourhood_group)

sns.boxplot(x="neighbourhood_group", y="price",  data=Airbnb_data, palette="Set2")
fig, ax = plt.subplots(figsize=(20,10))

box_picture = sns.boxplot(x="neighbourhood_group", y="price", hue='room_type', ax=ax, data=Airbnb_data)



for group in neighbourhood_group:

    temp = Airbnb_data.loc[Airbnb_data['neighbourhood_group'] == group]

    
count_data = pd.DataFrame()

nob_data = pd.DataFrame()



for group in neighbourhood_group:

    temp = Airbnb_data.loc[Airbnb_data['neighbourhood_group'] == group]

    count_data[group] = temp['price'].groupby(temp['room_type']).median()

    nob_data[group] = temp['price'].groupby(temp['room_type']).count()

nob_data
count_data
fig, ax = plt.subplots(figsize=(20,10))

sns.countplot(x='room_type', hue='neighbourhood_group', ax=ax, data=Airbnb_data)
select_data = Airbnb_data[['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365','price']]

sns.pairplot(select_data)
second_select_data = Airbnb_data[['minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'price']]

data = pd.concat([ second_select_data, pd.get_dummies(Airbnb_data['room_type']), pd.get_dummies(Airbnb_data['neighbourhood_group'])], axis=1)

data.sample(5)
from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBRegressor





train_x, test_x, train_y, test_y = train_test_split(data.drop('price', axis=1), data['price'], test_size=0.2, random_state=5)



# mymodle = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4, objective='reg:squarederror')

# mymodle.fit(train_x, train_y)

# prediction = mymodle.predict(test_x)





cv_params = {'n_estimators': range(400, 800, 100)}

other_params = {'learning_rate': 0.1, 'n_estimators': 600, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,

                   'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'objective' :'reg:squarederror'}



mymodle = XGBRegressor(**other_params)

gesearch = GridSearchCV(estimator=mymodle, param_grid=cv_params, scoring='r2', cv=5, n_jobs=4)

gesearch.fit(train_x, train_y)