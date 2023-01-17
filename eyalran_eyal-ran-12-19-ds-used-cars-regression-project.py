import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split as split

from sklearn.metrics import mean_squared_error as mse

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# For Kaggle Env run...

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

cars = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
cars.head()
# cars = pd.read_csv('vehicles.csv')
# cars.head()
print(cars.shape, '\n')
print(cars.info())
cars.isna().sum()
print('Number of duplicated rows according to "vin" column:', cars.vin.duplicated().sum())
print('Number of duplicated rows according to all columns:', cars.duplicated().sum())
print('Number of duplicated rows according to "vin" and "year" columns:', cars.duplicated(subset=['year', 'vin']).sum())
print('Number of duplicated rows according to "vin" and "id" columns:', cars.duplicated(subset=['id', 'vin']).sum())
print('Number of duplicated rows according to "vin", "id" and "year" columns:', cars.duplicated(subset=['id', 'year', 'vin']).sum())
print('Number of duplicated rows according to "id" column:', cars.id.duplicated().sum())
print('vin column nunique:', cars.vin.nunique(), '\n')
print('vin column value_counts:', cars.vin.value_counts())
print('id column nunique:', cars.id.nunique(), '\n')
print('id column value_counts:', cars.id.value_counts())
cars.dtypes.value_counts()
print('Numeric features:', '\n', cars.select_dtypes(exclude=object).columns, '\n')
print('Object features:', '\n', cars.select_dtypes(include=object).columns)
print('Number of Unique values in the price (target) feature:', cars.price.nunique(), '\n')
cars.price.describe()
cars.price.value_counts()
cars.price.loc[cars['price'] > 120000]
cars.price.loc[cars['price'] > 120000].value_counts()
def del_by_label(df, label_list, axis):
    return df.drop(labels=label_list, axis=axis)
print(cars.shape)
cars = del_by_label(cars, ['url', 'region_url', 'image_url', 'county', 'vin', 'id'], 1)
cars.shape
print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'price'] < 120].index), 0)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'price'] > 120000].index), 0)
cars.shape
# X_cars = cars.drop(labels='price', axis=1)
# y_cars = cars.price
# X_cars_train, X_cars_test, y_cars_train, y_cars_test = split(X_cars, y_cars, random_state=1)
# print('X_cars_train shape:', X_cars_train.shape)
# print('X_cars_test shape:', X_cars_test.shape)
# print('y_cars_train shape:', y_cars_train.shape)
# print('y_cars_testn shape:', y_cars_test.shape)
cars.head(10)
cars.shape
print('Number of Unique values in the price (target) feature:', cars.price.nunique(), '\n')
cars.price.describe()
cars.hist(column='price', grid=True, bins=80, range=(0, 80000), figsize=(10, 4))
print('Number of values above 50,000:', cars.price.loc[cars['price'] > 50000].count())
print('Number of values above 60,000:', cars.price.loc[cars['price'] > 60000].count())
print('Number of values above 70,000:', cars.price.loc[cars['price'] > 70000].count())
print('Number of values above 80,000:', cars.price.loc[cars['price'] > 80000].count())
sns.violinplot(x=cars.price)
fig, ax = plt.subplots(figsize=(22, 5))
ax.set_title('Number of Used Cars Ads per State')
sns.barplot(x=cars.state.value_counts().index, y=cars.state.value_counts().values, ax=ax)
print("Number of states included on 'state' column:", cars.state.nunique())
fig, ax = plt.subplots(figsize=(22, 5))
ax.set_title('Median Marked price per State')
sns.barplot(x=cars.state.value_counts().index, y=cars.groupby('state')['price'].median(), ax=ax)
price_per_location = {'state': cars.state, 'median_price_per_state': cars.groupby('state')['price'].transform('median'),
                      'region': cars.region, 'median_price_per_region': cars.groupby(['state', 'region'])['price'].transform('median'),
                      'lat': cars.lat, 'long': cars.long, 'median_price_per_lat_and_long': cars.groupby(['state', 'lat', 'long'])['price'].transform('median')
                     }

price_per_location = pd.DataFrame(price_per_location)
price_per_location
grouped_price_per_location = (price_per_location.groupby(['state', 'region', 'lat', 'long'])
                              ['median_price_per_state', 'median_price_per_region', 'median_price_per_lat_and_long'].mean())
grouped_price_per_location
grouped_price_per_location.loc['ny', :]
grouped_price_per_location.loc['ca', :]
lat_long_unique_count = grouped_price_per_location.groupby(['lat', 'long'])['median_price_per_lat_and_long'].count()
lat_long_unique_count
lat_long_unique_pairs_instances = lat_long_unique_count.value_counts()
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_title("'lat' and 'long' Unique Pairs Count")
ax.set_xlabel('Number of Unique Pairs Instances')
ax.set_ylabel('Unique Pairs')
sns.barplot(x=lat_long_unique_pairs_instances.index, y=lat_long_unique_pairs_instances, ax=ax)
print("Number of unique values in 'state' column:", cars.state.nunique())
print("Number of unique values in 'region' column:", cars.region.nunique())
print("Number of NaN values in 'year' column:", cars.year.isnull().sum())
cars.year.value_counts().to_frame().T
fig, ax = plt.subplots(figsize=(24, 4))
ax.set_title("Median Price per Year")
plt.xticks(rotation=90)
sns.barplot(x=cars.year, y=cars.groupby('year')['price'].transform('median'), ax=ax)
print("Number of NaN values in 'odometer' column:", cars.odometer.isnull().sum())
print('Number of cars with odometer value equal to zero:', cars.odometer[cars.odometer == 0].count())
print('Number of cars with odometer value equal to zero and condition value "new":',cars.loc[((cars.odometer == 0) & (cars.condition == 'new')), :].shape[0])
print('Number of cars with age value other than zero and condition value "new":', cars.loc[((cars.year != 2020) & (cars.condition == 'new')), :].shape[0])
print('Number of cars with age value other than zero and odometer value is zero:', cars.loc[((cars.odometer == 0) & (cars.year != 2020)), :].shape[0])
print('Number of cars with age value other than zero and odometer value is zero:', cars.loc[((cars.odometer == 0) & (cars.year < 1988)), :].shape[0])
cars.hist(column='odometer', grid=True, bins=100, range=(0, 400000), figsize=(10, 4))
print('Number of cars with odometer value above 400,000:', cars.odometer[cars.odometer > 400000].count())
print('Number of cars with odometer value less than 100:', cars.odometer[cars.odometer < 10].count())
cars.loc[cars.year > 1987.00, :].groupby('year')['odometer'].value_counts()
cars[['year', 'odometer']].corr()
cars.groupby('manufacturer')['price'].mean().sort_values().plot(kind='barh', figsize=(11, 11), legend='True', color='r')
cars.groupby('model')['price'].mean().sort_values()
cars.groupby('model')['price'].mean().sort_values().plot(figsize=(12, 4))
cars.groupby(['manufacturer', 'model'])['price'].mean().to_frame().T
print(cars.type.value_counts(dropna=False))
print(cars['size'].value_counts(dropna=False))
print(cars.condition.value_counts(dropna=False))
sns.barplot(x=cars.loc[cars.year > 1987, :].condition, y=cars.loc[cars.year > 1987, :].price)
cars.loc[cars.year > 1987, :].groupby('condition')['year'].mean()
print(cars.title_status.value_counts(dropna=False))
print(cars.cylinders.value_counts(dropna=False))
cars.groupby('cylinders')['fuel'].value_counts()
cars.groupby('cylinders')['transmission'].value_counts()
print(cars.fuel.value_counts(dropna=False))
cars.groupby('fuel')['type'].value_counts(dropna=False).head(30)
print(cars.transmission.value_counts(dropna=False))
print(cars.drive.value_counts(dropna=False))
sns.barplot(x=cars.loc[cars.year > 1987, :].drive, y=cars.loc[cars.year > 1987, :].price)
cars.groupby('drive')['type'].value_counts(dropna=False).head(30)
print(cars.paint_color.value_counts(dropna=False))
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=cars.loc[cars.year > 1987, :].paint_color, y=cars.loc[cars.year > 1987, :].price)
cars.loc[0, 'description']
cars.loc[2, 'description']
cars.loc[10000, 'description']
print(cars.shape)
cars = del_by_label(cars, ['model', 'size', 'state', 'lat', 'long'], 1)
cars.shape
print(cars.shape)
cars = del_by_label(cars, 'condition', 1)
cars.shape
print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'year'] < 1988].index), 0)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'year'] == 2021].index), 0)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'year'] == 0].index), 0)
cars.shape
print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'title_status'] == 'salvage'].index), 0)
cars.shape
print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'title_status'] == 'missing'].index), 0)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'title_status'] == 'parts only'].index), 0)
cars.shape
print(cars.shape)
cars = del_by_label(cars, (cars.loc[cars.loc[:, 'cylinders'] == 'other'].index), 0)
cars.shape
print(cars.shape)
cars = cars.dropna(axis=0, subset=['odometer', 'year'])
cars.shape
cars.year = cars.year.replace(to_replace={0: 1})
cars.insert(loc=7, column='milage_per_year', value=(cars.odometer / (2020 - cars.year)), allow_duplicates=True)
cars.milage_per_year = cars.milage_per_year.fillna(value=0)
cars.milage_per_year = cars.milage_per_year.replace(to_replace={np.inf: 0, -np.inf: 0})
cars.insert(loc=2, column='age', value=(2020 - cars.year), allow_duplicates=True)
cars = del_by_label(cars, 'year', 1)
color_dict = {'white': 'white_black', 'black': 'white_black', 'silver': 'other', 'blue': 'other', 'red': 'other',
              'grey': 'other', 'green': 'other', 'custom': 'other', 'brown': 'other', 'orange': 'other', 'yellow': 'other',
              'purple': 'other', np.nan: 'other'}
cars.paint_color = cars.paint_color.replace(color_dict)
cylinders_dict = {'6 cylinders': 6, '8 cylinders': 8, '4 cylinders': 4, '5 cylinders': 5, '10 cylinders': 10,
              '3 cylinders': 3, '12 cylinders': 12, np.nan: 'other'}
cars.cylinders = cars.cylinders.replace(cylinders_dict)
fuel_dict = {'gas': 'gas', 'diesel': 'other', 'other': 'other', 'hybrid': 'other', 'electric': 'other', np.nan: 'other'}
cars.fuel = cars.fuel.replace(fuel_dict)
title_dict = {'clean': 'clean', 'rebuilt': 'other', 'lien': 'other', np.nan: 'other'}
cars.title_status = cars.title_status.replace(title_dict)
transmission_dict = {'automatic': 'automatic', 'manual': 'other', 'other': 'other', np.nan: 'other'}
cars.transmission = cars.transmission.replace(transmission_dict)
cars.drive.value_counts()
cars.drive = cars.drive.fillna(value='other')
cars.drive.value_counts()
cars = pd.get_dummies(data=cars, columns=['fuel', 'title_status', 'transmission', 'drive', 'paint_color'])
description_col = cars.description
print(cars.shape)
cars = del_by_label(cars, ['description'], 1)
cars.shape
description_col.str.contains(pat='^V[0-9]|^V[0-9][0-9]', case=False, regex=True).sum()
cars['log_price'] = np.log1p(cars.price)
cars = del_by_label(cars, 'price', 1)
cars.odometer = cars.odometer / 100
cars.milage_per_year = cars.milage_per_year / 100
cars = cars.reset_index(drop=True)
X_cars = cars.drop(labels='log_price', axis=1)
y_cars = cars.log_price
X_cars_train, X_cars_test, y_cars_train, y_cars_test = split(X_cars, y_cars, random_state=1)
print('X_cars_train shape:', X_cars_train.shape)
print('X_cars_test shape:', X_cars_test.shape)
print('y_cars_train shape:', y_cars_train.shape)
print('y_cars_testn shape:', y_cars_test.shape)
class TargetMedianPerValTransformer():  
    
    def __init__(self, val_count, exclude=None):
        self.val_count = val_count
        self.exclude_list = exclude
        self.fit_dict = dict()
    
    def col_labels_for_fit(self, X):
        label_mask = ((X.nunique() > self.val_count) & (X.dtypes == object))
        for label in self.exclude_list:
            label_mask[label] = False
        return label_mask[label_mask].index
    
    def label_dict_generator(self, X, y):
        for label in self.fitted_feature_labels:
            if X[label].isna().any():
                X[label] = X[label].replace(to_replace=np.nan, value='other')
            grouper = X.groupby(label)
            self.fit_dict[label] = dict()
            for name, group in grouper:
                group_indices = grouper.get_group(name).index
                self.fit_dict[label][name] = y[group_indices].mean()
    
    def fit(self, X, y):
        self.fitted_feature_labels = self.col_labels_for_fit(X)
        self.label_dict_generator(X, y)
        return self
    
    def transform(self, X):
        for label in self.fitted_feature_labels:
            X[label] = X[label].map(self.fit_dict[label])
            if X[label].isna().any():
                X[label] = X[label].where(cond=(X[label] == np.nan), other=(X[label].median()))
        return X
transform_multy_val_car_features = TargetMedianPerValTransformer(val_count=4, exclude=['description'])
transform_multy_val_car_features.fit(X_cars_train, y_cars_train)
X_cars_train = transform_multy_val_car_features.transform(X_cars_train)
X_cars_test = transform_multy_val_car_features.transform(X_cars_test)
# X_cars_train = X_cars_train.astype(dtype={'region': np.float32, 'age': np.float32,
#                    'manufacturer': np.float32, 'cylinders': np.float32,
#                    'odometer': np.float32, 'milage_per_year': np.float32,
#                    'type': np.float32})

# X_cars_test = X_cars_test.astype(dtype={'region': np.float32, 'age': np.float32,
#                    'manufacturer': np.float32, 'cylinders': np.float32,
#                    'odometer': np.float32, 'milage_per_year': np.float32,
#                    'type': np.float32})
X_cars_train.info()
X_cars_train.describe()
X_cars_test.describe()
X_cars_train.head(10)
cars_tree_model = DecisionTreeRegressor(criterion='mse',
                                        splitter='random',
                                        min_samples_split=7000,
                                        min_samples_leaf=2600)
cars_tree_model.fit(X_cars_train, y_cars_train)
print(cars_tree_model.max_features_)
print(cars_tree_model.n_features_)
print(cars_tree_model.n_outputs_)
print(cars_tree_model.feature_importances_)
for feature, importance in zip(X_cars_train.columns, cars_tree_model.feature_importances_):
    print(f'{feature:12}: {importance}')
!pip install pydot
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
def visualize_tree(model, md=10, width=1000):
    dot_data = StringIO()  
    export_graphviz(model, out_file=dot_data, feature_names=X_cars_train.columns, max_depth=md)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]  
    return Image(graph.create_png(), width=width)
# visualize_tree(cars_tree_model, md=12, width=1200)
# plot_tree(decision_tree=cars_tree_model, filled=True)
y_train_tree_pred = cars_tree_model.predict(X_cars_train)
tree_train_pred_dict = {'y_true': y_cars_train, 'y_pred': y_train_tree_pred}
pd.DataFrame(tree_train_pred_dict)
ax = sns.scatterplot(x=y_cars_train, y=y_train_tree_pred)
ax.plot(y_cars_train, y_cars_train, 'r')
RMSE_train = mse(y_cars_train, y_train_tree_pred)**0.5
RMSE_train
y_test_tree_pred = cars_tree_model.predict(X_cars_test)
tree_test_pred_dict = {'y_true': y_cars_test, 'y_pred': y_test_tree_pred}
pd.DataFrame(tree_test_pred_dict)
RMSE_test = mse(y_cars_test, y_test_tree_pred)**0.5
RMSE_test