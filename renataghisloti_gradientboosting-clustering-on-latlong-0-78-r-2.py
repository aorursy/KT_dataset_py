import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error



%matplotlib inline
df = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
df.head()
df['median_house_value'].hist(bins=40)
df.shape
ax = df.hist(figsize=[20, 18], bins = 50)
ax = pd.plotting.scatter_matrix(df, alpha=0.2, figsize = [20, 20])
df['peopleperrooms'] = df['households']/df['total_rooms']
df['latlon'] = df['longitude'] * df['latitude']
df['roomsperbedrooms'] = df['total_rooms']/df['total_bedrooms']
df['peopleperbedrooms'] = df['households']/df['total_bedrooms']
df['ageperpop'] = df['housing_median_age']/df['households']
df['ageperbeds'] = df['housing_median_age']/df['total_bedrooms']
list(df.columns.values)
df.head()
ax = pd.plotting.scatter_matrix(df[['median_house_value', 

                                    'peopleperrooms', 

                                    'latlon', 

                                    'roomsperbedrooms',

                                    'peopleperbedrooms',

                                    'ageperpop',

                                    'ageperbeds']], figsize = [32, 28])
df[df.isnull().any(axis=1)]
df = df[~df.isnull().any(axis=1)]
plt.scatter(df['longitude'], df['latitude'], marker = ".")
X = df[['longitude','latitude']]

means = KMeans(n_clusters=60, random_state=0).fit(X)
means.cluster_centers_[0:5]
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(df['longitude'], df['latitude'], marker = ".")

centroids = means.cluster_centers_



ax.scatter(centroids[:, 0], centroids[:, 1], color = "black", marker = ",")
df['geo_cluster'] = means.predict(X)
df.groupby("geo_cluster")['median_house_value'].mean().head()
df.groupby("geo_cluster")['median_house_value'].mean().sort_values().head()
index = df.groupby("geo_cluster")['median_house_value'].mean().sort_values().index

new_index = range(0, 60)

dict_index = dict(zip(index, new_index))

dict_index
df['geo_cluster_ordered'] = df['geo_cluster'].apply(lambda x: dict_index[x])
df.head()
df.groupby('ocean_proximity')['median_house_value'].mean()
values = df.groupby('ocean_proximity')['median_house_value'].mean()

new_values = dict(zip(values.index, values.values))

new_values
df['n_ocean_proximity'] = df['ocean_proximity'].apply(lambda x: new_values[x])
ax = pd.plotting.scatter_matrix(df[['median_house_value', 'n_ocean_proximity', 'geo_cluster_ordered']], alpha=0.8, figsize = [40, 36])
X = df.drop(columns=['median_house_value', 'ocean_proximity', 'latlon', 'geo_cluster'])

y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = GradientBoostingRegressor().fit(X_train, y_train)
y_pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred)))
r2_score(y_test, y_pred)
mean_absolute_error(y_test, y_pred)