# import python standard library

import math



# import data manipulation library

import numpy as np

import pandas as pd



# import data visualization library

import matplotlib.pyplot as plt

import seaborn as sns



# import model function from sklearn

from sklearn.ensemble import RandomForestRegressor



# import sklearn model selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



# import sklearn model evaluation regression metrics

from sklearn.metrics import mean_squared_error
# acquiring training and testing data

df_train = pd.read_csv('../input/train.csv', nrows=2000000, parse_dates=['pickup_datetime'])

df_test = pd.read_csv('../input/test.csv', parse_dates=['pickup_datetime'])
# visualize head of the training data

df_train.head(n=3)
# visualize tail of the testing data

df_test.tail(n=3)
# combine training and testing dataframe

df_train['datatype'], df_test['datatype'] = 'training', 'testing'

df_test.insert(1, 'fare_amount', 0)

df_data = pd.concat([df_train, df_test], ignore_index=True)
def scatterplot(numerical_x: list or str, numerical_y: list or str, data: pd.DataFrame, figsize: tuple = (4, 3), ncols: int = 5, nrows: int = None) -> plt.figure:

    """ Return a scatter plot applied for numerical variable in x-axis vs numerical variable in y-axis.

    

    Args:

        numerical_x (list or str): The numerical variable in x-axis.

        numerical_y (list or str): The numerical variable in y-axis.

        data (pd.DataFrame): The data to plot.

        figsize (tuple): The matplotlib figure size width and height in inches. Default to (4, 3).

        ncols (int): The number of columns for axis in the figure. Default to 5.

        nrows (int): The number of rows for axis in the figure. Default to None.

    

    Returns:

        plt.figure: The plot figure.

    """

    

    numerical_x, numerical_y = [numerical_x] if type(numerical_x) == str else numerical_x, [numerical_y] if type(numerical_y) == str else numerical_y

    if nrows is None: nrows = (len(numerical_x)*len(numerical_y) - 1) // ncols + 1

    

    fig, axes = plt.subplots(figsize=(figsize[0]*ncols , figsize[1]*nrows), ncols=ncols, nrows=nrows)

    axes = axes.flatten()

    _ = [sns.scatterplot(x=vj, y=vi, data=data, ax=axes[i*len(numerical_x) + j], rasterized=True) for i, vi in enumerate(numerical_y) for j, vj in enumerate(numerical_x)]

    return fig
def distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:

    """ Return the distance between 2 points of latitude and longitude.

    

    Args:

        lat1 (float): The latitude of the first coordinate.

        lon1 (float): The longitude of the first coordinate.

        lat2 (float): The latitude of the second coordinate.

        lon2 (float): The longitude of the second coordinate.

    

    Returns:

        float: The distance between 2 points of latitude and longitude.

    """

    angle = 0.017453292519943295 #math.pi / 180

    x = 0.5 - np.cos((lat2 - lat1) * angle) / 2 + np.cos(lat1 * angle) * np.cos(lat2 * angle) * (1 - np.cos((lon2 - lon1) * angle)) / 2

    return 0.6213712 * 12742 * np.arcsin(np.sqrt(x))
# describe training and testing data

df_data.describe(include='all')
# list all features type number

col_number = df_data.select_dtypes(include=['number']).columns.tolist()

print('features type number:\n items %s\n length %d' %(col_number, len(col_number)))



# list all features type object

col_object = df_data.select_dtypes(include=['object']).columns.tolist()

print('features type object:\n items %s\n length %d' %(col_object, len(col_object)))
# feature exploration: histogram of all numeric features

_ = df_data.hist(bins=20, figsize=(20, 15))
# feature extraction: fare amount

df_data['fare_amount'] = np.log1p(df_data['fare_amount'])
# feature extraction: combination of keyword date

df_data['year'] = df_data['pickup_datetime'].dt.year

df_data['quarter'] = df_data['pickup_datetime'].dt.quarter

df_data['month'] = df_data['pickup_datetime'].dt.month

df_data['weekofyear'] = df_data['pickup_datetime'].dt.weekofyear

df_data['weekday'] = df_data['pickup_datetime'].dt.weekday

df_data['dayofweek'] = df_data['pickup_datetime'].dt.dayofweek

df_data['hour'] = df_data['pickup_datetime'].dt.hour
# feature extraction: distance

df_data['distance_euclidean'] = distance(df_data['pickup_latitude'], df_data['pickup_longitude'], \

                                         df_data['dropoff_latitude'], df_data['dropoff_longitude'])

df_data['distance_latitude'] = df_data['dropoff_latitude'] - df_data['pickup_latitude']

df_data['distance_longitude'] = df_data['dropoff_longitude'] - df_data['pickup_longitude']
# feature extraction: distance to specific location

nyc = (40.7128, -74.0060)

jfk = (40.6413, -73.7781)

ewr = (40.6895, -74.1745)

df_data['distance_pickup_to_nyc'] = distance(df_data['pickup_latitude'], df_data['pickup_longitude'], nyc[0], nyc[1])

df_data['distance_pickup_to_jfk'] = distance(df_data['pickup_latitude'], df_data['pickup_longitude'], jfk[0], jfk[1])

df_data['distance_pickup_to_ewr'] = distance(df_data['pickup_latitude'], df_data['pickup_longitude'], ewr[0], ewr[1])

df_data['distance_dropoff_to_nyc'] = distance(df_data['dropoff_latitude'], df_data['dropoff_longitude'], nyc[0], nyc[1])

df_data['distance_dropoff_to_jfk'] = distance(df_data['dropoff_latitude'], df_data['dropoff_longitude'], jfk[0], jfk[1])

df_data['distance_dropoff_to_ewr'] = distance(df_data['dropoff_latitude'], df_data['dropoff_longitude'], ewr[0], ewr[1])
# feature extraction: fare amount per mile

df_data['fare_per_mile'] = df_data['fare_amount'] / df_data['distance_euclidean']

df_data['fare_per_mile'] = df_data['fare_per_mile'].apply(lambda x: 0 if x == float('inf') else x)

df_data['fare_per_mile'] = df_data['fare_per_mile'].fillna(0)
# feature exploration: fare amount

col_number = df_data.select_dtypes(include=['number']).columns.tolist()

_ = scatterplot(col_number, 'fare_amount', df_data[df_data['datatype'] == 'training'])
# feature exploration: fare per mile

col_number = df_data.select_dtypes(include=['number']).columns.tolist()

_ = scatterplot(col_number, 'fare_per_mile', df_data[df_data['datatype'] == 'training'])
# feature exploration: season dataframe

df_season = df_data[df_data['datatype'] == 'training'].groupby(['year', 'month'], as_index=False).agg({

    'fare_amount': 'mean'

})

fig, axes = plt.subplots(figsize=(20, 3))

_ = sns.pointplot(x='month', y='fare_amount', data=df_season, join=True, hue='year')
# feature exploration: season dataframe

df_season = df_data[df_data['datatype'] == 'training'].groupby(['year', 'hour'], as_index=False).agg({

    'fare_amount': 'mean'

})

fig, axes = plt.subplots(figsize=(20, 3))

_ = sns.pointplot(x='hour', y='fare_amount', data=df_season, join=True, hue='year')
# feature extraction: drop na

df_data = df_data.dropna()
# convert category codes for data dataframe

df_data = pd.get_dummies(df_data, columns=['datatype'], drop_first=True)
# describe data dataframe

df_data.describe(include='all')
# verify dtypes object

df_data.info()
# compute pairwise correlation of columns, excluding NA/null values and present through heat map

corr = df_data[df_data['datatype_training'] == 1].drop(['key'], axis=1).corr()

fig, axes = plt.subplots(figsize=(20, 15))

heatmap = sns.heatmap(corr, annot=True, cmap=plt.cm.RdBu, fmt='.1f', square=True, vmin=-0.8, vmax=0.8)
# select all features

x = df_data[df_data['datatype_training'] == 1].drop(['key', 'pickup_datetime', 'fare_amount', 'fare_per_mile', 'datatype_training'], axis=1)

y = df_data[df_data['datatype_training'] == 1]['fare_amount']
# perform train-test (validate) split

x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.25, random_state=58)
# random forest regression model setup

model_forestreg = RandomForestRegressor(n_estimators=10, max_depth=20, min_samples_split=1000, random_state=58)



# random forest regression model fit

model_forestreg.fit(x_train, y_train)



# random forest regression model prediction

model_forestreg_ypredict = model_forestreg.predict(x_validate)



# random forest regression model metrics

model_forestreg_rmse = mean_squared_error(y_validate, model_forestreg_ypredict) ** 0.5

model_forestreg_cvscores = np.sqrt(np.abs(cross_val_score(model_forestreg, x, y, cv=5, scoring='neg_mean_squared_error')))

print('random forest regression\n  root mean squared error: %0.4f, cross validation score: %0.4f (+/- %0.4f)' %(model_forestreg_rmse, model_forestreg_cvscores.mean(), 2 * model_forestreg_cvscores.std()))
# model selection

final_model = model_forestreg



# prepare testing data and compute the observed value

x_test = df_data[df_data['datatype_training'] == 0].drop(['key', 'pickup_datetime', 'fare_amount', 'fare_per_mile', 'datatype_training'], axis=1)

y_test = pd.DataFrame(np.expm1(final_model.predict(x_test)), columns=['fare_amount'], index=df_data.loc[df_data['datatype_training'] == 0, 'key'])
# submit the results

out = pd.DataFrame({'key': y_test.index, 'fare_amount': y_test['fare_amount']})

out.to_csv('submission.csv', index=False)