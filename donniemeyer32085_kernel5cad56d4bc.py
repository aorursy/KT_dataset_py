import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, KFold
df = pd.read_csv('../input/real_estate_data.csv')
df.head()
df.info()
df = df.drop(['No', 'X5 latitude', 'X6 longitude', 'X1 transaction date'], axis=1)
columns = ['house_age',

       'distance_to _the_nearest_MRT_station',

       'number_of_convenience_stores', 'house_price_of_unit_area']
df.columns = columns
df.isnull().sum()
normalized_df = (df - df.mean()) / (df.std())                   

normalized_df['house_price_of_unit_area'] = df['house_price_of_unit_area']

normalized_df.head()
for col in columns:

    if col != 'house_price_of_unit_area':

        plt.scatter(normalized_df[col], normalized_df['house_price_of_unit_area'])

        plt.show()

    else:

        continue
X = normalized_df[['house_age', 'distance_to _the_nearest_MRT_station', 'number_of_convenience_stores']]  #independent columns

y = normalized_df['house_price_of_unit_area']   #target column i.e price range

#get correlations of each features in dataset

corrmat = normalized_df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(10,7))

#plot heat map

g=sns.heatmap(normalized_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
print(len(normalized_df) * 0.75)

print(len(normalized_df) * 0.25)
traing_data = normalized_df.iloc[0:311]

test_data = normalized_df.iloc[311:]
feature_cols = ['house_age', 'distance_to _the_nearest_MRT_station', 'number_of_convenience_stores']
traing_data.head()
test_data.head()
knn = KNeighborsRegressor()

knn.fit(traing_data[feature_cols], traing_data['house_price_of_unit_area'])

predictions = knn.predict(test_data[feature_cols])

rmse = np.sqrt(mean_squared_error(test_data['house_price_of_unit_area'], predictions))

print(rmse)
knn = KNeighborsRegressor()

knn.fit(traing_data[['distance_to _the_nearest_MRT_station', 'number_of_convenience_stores']], traing_data['house_price_of_unit_area'])

predictions = knn.predict(test_data[['distance_to _the_nearest_MRT_station', 'number_of_convenience_stores']])

rmse = np.sqrt(mean_squared_error(test_data['house_price_of_unit_area'], predictions))

print(rmse)
knn = KNeighborsRegressor()

knn.fit(traing_data[['distance_to _the_nearest_MRT_station']], traing_data['house_price_of_unit_area'])

predictions = knn.predict(test_data[['distance_to _the_nearest_MRT_station']])

rmse = np.sqrt(mean_squared_error(test_data['house_price_of_unit_area'], predictions))

print(rmse)
hyper_params = list(range(1,21))

print(hyper_params)
rmse_list = []

for hyper_param in hyper_params:

    knn = KNeighborsRegressor(n_neighbors=hyper_param)

    knn.fit(traing_data[feature_cols], traing_data['house_price_of_unit_area'])

    predictions = knn.predict(test_data[feature_cols])

    rmse = np.sqrt(mean_squared_error(test_data['house_price_of_unit_area'], predictions))

    rmse_list.append(rmse)

print(rmse_list)
plt.scatter(hyper_params, rmse_list)
rmse_list = []

for hyper_param in hyper_params:

    knn = KNeighborsRegressor(n_neighbors=hyper_param)

    knn.fit(traing_data[['distance_to _the_nearest_MRT_station', 'number_of_convenience_stores']], traing_data['house_price_of_unit_area'])

    predictions = knn.predict(test_data[['distance_to _the_nearest_MRT_station', 'number_of_convenience_stores']])

    rmse = np.sqrt(mean_squared_error(test_data['house_price_of_unit_area'], predictions))

    rmse_list.append(rmse)

print(rmse_list)
plt.scatter(hyper_params, rmse_list)
kf = KFold(n_splits=3, shuffle=True, random_state=1)



knn = KNeighborsRegressor()



mses = cross_val_score(knn, X=traing_data[feature_cols], y=traing_data['house_price_of_unit_area'], scoring='neg_mean_squared_error', cv=kf)

print(mses)

rmses = np.sqrt(np.absolute(mses))

avg_rmse = np.mean(rmses)



print(rmses)

print(avg_rmse)
kf = KFold(n_splits=8, shuffle=True, random_state=1)



knn = KNeighborsRegressor()



mses = cross_val_score(knn, X=traing_data[['distance_to _the_nearest_MRT_station', 'number_of_convenience_stores']], y=traing_data['house_price_of_unit_area'], scoring='neg_mean_squared_error', cv=kf)

print(mses)

rmses = np.sqrt(np.absolute(mses))

avg_rmse = np.mean(rmses)



print(rmses)

print(avg_rmse)