import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
nyc_airbnb = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

nyc_airbnb.head()
print("The number of records(examples): {}".format(nyc_airbnb.shape[0]))
print("The number of columns(features): {}".format(nyc_airbnb.shape[1]))
# checking type of every column in the dataset
nyc_airbnb.dtypes
print("Null values in NYC Airbnb 2019 dataset:")
# checking total missing values in each column in the dataset
nyc_airbnb.isnull().sum()
# https://numpy.org/doc/1.18/reference/generated/numpy.where.html
# numpy.where: Return elements chosen from x or y depending on condition.
null_names = pd.DataFrame(np.where(nyc_airbnb['name'].isnull())).transpose()
null_host_names = pd.DataFrame(np.where(nyc_airbnb['host_name'].isnull())).transpose()

concat_null_names = pd.concat([null_names, null_host_names], axis=1, ignore_index=True)
concat_null_names.columns = ['Null rows in name column', 'Null rows in host_name column']
concat_null_names
import missingno as msno
missing_value_columns = nyc_airbnb.columns[nyc_airbnb.isnull().any()].tolist()
print("Missing value columns: {}".format(missing_value_columns))
msno.bar(nyc_airbnb[missing_value_columns], figsize=(15,8), color='#2A3A7E', 
         fontsize=15, labels=True)  # Can switch to a logarithmic scale by specifying log=True
msno.matrix(nyc_airbnb[missing_value_columns], width_ratios=(10, 1),
            figsize=(20, 8), color=(0, 0, 0), fontsize=12, sparkline=True, labels=True)
nyc_airbnb.drop(['last_review'], axis=1, inplace=True)
nyc_airbnb.drop(['host_name'], axis=1, inplace=True)
nyc_airbnb.head(5)
nyc_airbnb['name'].fillna(value=0, inplace=True)
nyc_airbnb['reviews_per_month'].fillna(value=0, inplace=True)
nyc_airbnb.isnull().sum()
# check unique category values
# find out which neighbourhood_group exist in dataset
print('Neighbourhood_group: {}'.format(nyc_airbnb['neighbourhood_group'].unique()))
# check unique category values
# find out which neighbourhood exist in dataset
nyc_airbnb['neighbourhood'].unique()
nyc_airbnb['neighbourhood'].nunique()
nyc_airbnb['room_type'].unique()
nyc_airbnb.sort_values(by='number_of_reviews', ascending=False).head(5)
nyc_airbnb.sort_values(by='reviews_per_month', ascending=False).head(5)
plt.figure(figsize=(15, 8))
plt.title('Counts of airbnb in neighbourhood group', fontsize=15)
sns.countplot(x='neighbourhood_group', data=nyc_airbnb, 
              order=nyc_airbnb['neighbourhood_group'].value_counts().index,
              palette='BuGn_r')
plt.figure(figsize=(15, 8))
plt.title('Counts of airbnb in neighbourhood group with room type', fontsize=15)
sns.countplot(x='neighbourhood_group', data=nyc_airbnb, hue='room_type',
              palette="Set2")
top_neigh = nyc_airbnb['neighbourhood'].value_counts().reset_index().head(10)  # Top 10
top_neigh = top_neigh['index'].tolist()  # get top 10 neighbourhood names

plt.figure(figsize=(15, 8))
plt.title('Top neighbourhoods with room type', fontsize=15)
viz = sns.countplot(x='neighbourhood', data=nyc_airbnb.loc[nyc_airbnb['neighbourhood'].isin(top_neigh)],
              hue='room_type', palette='GnBu_d')
viz.set_xticklabels(viz.get_xticklabels(), rotation=45)
# check wholde dataset price stats
nyc_airbnb['price'].describe()
plt.figure(figsize=(15, 8))
sns.distplot(nyc_airbnb['price'])
nyc_airbnb['price'].quantile(.98)
plt.figure(figsize=(15, 8))
sns.distplot(nyc_airbnb[nyc_airbnb['price'] < 550]['price'])
plt.figure(figsize=(15, 8))
plt.title('Density and distribution of prices for each neighbourhood group', fontsize=15)
sns.violinplot(x='neighbourhood_group', y='price', 
               data=nyc_airbnb[nyc_airbnb['price'] < 550], palette='Set3')
# Brooklyn
sub_1_brooklyn = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Brooklyn']
price_sub_1 = sub_1_brooklyn[['price']]

# Manhattan
sub_2_manhattan = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Manhattan']
price_sub_2 = sub_2_manhattan[['price']]

# Queeens
sub_3_queens = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Queens']
price_sub_3 = sub_3_queens[['price']]

# Staten Island
sub_4_staten = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Staten Island']
price_sub_4 = sub_4_staten[['price']]

# Bronx
sub_5_bronx = nyc_airbnb.loc[nyc_airbnb['neighbourhood_group'] == 'Bronx']
price_sub_5 = sub_5_bronx[['price']]

price_list_by_group = [price_sub_1, price_sub_2, price_sub_3, price_sub_4, price_sub_5]
integ_price_stats_list = []
neigh_groups = nyc_airbnb['neighbourhood_group'].unique().tolist()

for price_group, group_name in zip(price_list_by_group, neigh_groups):
  stats = price_group.describe()  # count / mean / std / 25% / 50% / 75% / max
  stats = stats.iloc[1:]  # mean / std / 25% / 50% / 75% / max
  stats.reset_index(inplace=True)
  stats.rename(columns={'index': 'Stats', 'price': group_name}, inplace=True)
  stats.set_index('Stats', inplace=True)
  integ_price_stats_list.append(stats)

price_stats_df = pd.concat(integ_price_stats_list, axis=1)
price_stats_df
cmap = sns.cubehelix_palette(as_cmap=True)

wo_extreme = nyc_airbnb[nyc_airbnb['price'] < 550]

f, ax = plt.subplots()
f.set_size_inches(20, 10)
points = ax.scatter(wo_extreme['latitude'], wo_extreme['longitude'], 
                    c=wo_extreme['price'], cmap=cmap)
f.colorbar(points)
plt.figure(figsize=(15, 8))
sns.scatterplot(data=nyc_airbnb, x='longitude', y='latitude', 
                hue='neighbourhood_group', palette='Set3')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
nyc_airbnb.drop(['name', 'id'], inplace=True, axis=1)
# encodes categorical values
le = LabelEncoder()

le.fit(nyc_airbnb['neighbourhood_group'])
nyc_airbnb['neighbourhood_group'] = le.transform(nyc_airbnb['neighbourhood_group'])

le.fit(nyc_airbnb['neighbourhood'])
nyc_airbnb['neighbourhood'] = le.transform(nyc_airbnb['neighbourhood'])

le.fit(nyc_airbnb['room_type'])
nyc_airbnb['room_type'] = le.transform(nyc_airbnb['room_type'])

nyc_airbnb.head(5)
# records with price zero are sorted on top
nyc_airbnb.sort_values('price', ascending=True, inplace=True)
nyc_airbnb = nyc_airbnb[11:-6]
lm = LinearRegression()
X = nyc_airbnb.drop(['price', 'longitude'], inplace=False, axis=1)
y = nyc_airbnb['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
# Evaluated metrics

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
r2 = metrics.r2_score(y_test, predictions)

print('MAE (Mean Absolute Error): %s' %mae)
print('MSE (Mean Squared Error): %s' %mse)
print('RMSE (Root mean squared error): %s' %rmse)
print('R2 score: %s' %r2)
# Avtual vs predicted values

error = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': predictions.flatten()})
error.head(10)