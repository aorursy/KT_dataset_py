import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error



from time import time



from warnings import filterwarnings

filterwarnings(action='ignore')



sns.set_style('whitegrid')

sns.set_palette('viridis')



%matplotlib inline
df_train = pd.read_csv('../input/train.csv', nrows=50_000)

df_test = pd.read_csv('../input/test.csv')
df_train.info()
nan_train = pd.DataFrame(data=df_train.isnull().sum(), columns=['Train NaN'])

nan_test = pd.DataFrame(data=df_test.isnull().sum(), columns=['Test NaN'])

nan_test.loc['fare_amount'] = 0

pd.concat([nan_train, nan_test], axis=1, sort=False)
df_train.dropna(inplace=True)
df_train.describe()
def bouding_box(df):            

        # Bounding box

        latitude_min, latitude_max = (40.4774, 40.9162)

        longitude_min, longitude_max = (-74.2591, -73.7002)

        # Applying the limits

        true_coordinates = df['pickup_latitude'].between(latitude_min, latitude_max)

        true_coordinates &= df['pickup_longitude'].between(longitude_min, longitude_max)

        true_coordinates &= df['dropoff_latitude'].between(latitude_min, latitude_max)

        true_coordinates &= df['dropoff_longitude'].between(longitude_min, longitude_max)

        return df[true_coordinates]
df_train = bouding_box(df_train)
def clear_fare(df):

    

    # Fare interval

    min_fare, max_fare = 1.50, 100

    # Applying the limits

    true_fare = df['fare_amount'].between(min_fare, max_fare)

    return df[true_fare]
df_train = clear_fare(df_train)
def split_datetime(df):          

    # Split datetime column

    datetime = pd.to_datetime(df['pickup_datetime'])

    df['day_of_week'] = datetime.dt.dayofweek

    df['day_of_month'] = datetime.dt.day

    df['month'] = datetime.dt.month

    df['year'] = datetime.dt.year

    df['hour'] = datetime.dt.hour + datetime.dt.minute/60

    return df
df_train = split_datetime(df_train)

df_test = split_datetime(df_test)
def haversine(coordinates):

    

    from math import pi, sqrt, sin, cos, atan2

    

    lat1 = coordinates[0]

    long1 =  coordinates[1]

    lat2 = coordinates[2]

    long2 = coordinates[3]



    degree_to_rad = float(pi / 180.0)



    d_lat = (lat2 - lat1) * degree_to_rad

    d_long = (long2 - long1) * degree_to_rad



    a = pow(sin(d_lat / 2), 2) + cos(lat1 * degree_to_rad) * cos(lat2 * degree_to_rad) * pow(sin(d_long / 2), 2)

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    km = 6367 * c



    return km



def distance(df):

    # Compute the amount of latitude and longitude deslocation

    df['delta_latitude'] = (df['dropoff_latitude'] - df['pickup_latitude'])

    df['delta_longitude'] = (df['dropoff_longitude'] - df['pickup_longitude'])

    # Compute the amount of displacement

    #bs: I'm treating angles as 2D plane coordinates to derive the following feature

    #df['displacement_degree'] = np.linalg.norm(df[['delta_latitude', 'delta_longitude']], axis=1)

    df['distance_km'] = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']].apply(haversine, axis=1, raw=True)    

    return df
df_train = distance(df_train)

df_test = distance(df_test)
def move_directions(df):

    

    new_df = pd.DataFrame()

    # Creates a column with true values for travel going north

    new_df['north'] = ((df['delta_latitude'] > 0) & (df['distance_km'] >= 0.1)).astype('int')

    # Creates a column with true values for travel going south

    new_df['south'] = ((df['delta_latitude'] < 0) & (df['distance_km'] >= 0.1)).astype('int')

    # Creates a column with true values for travel going west

    new_df['west'] = ((df['delta_longitude'] < 0) & (df['distance_km'] >= 0.1)).astype('int')

    # Creates a column with true values for travel going east

    new_df['east'] = ((df['delta_longitude'] > 0) & (df['distance_km'] >= 0.1)).astype('int')

    # Creates a column with true values for travel that start and finish at the same point

    new_df['unknown'] = (df['distance_km'] <= 0.1).astype('int')    

    return new_df



def wind_rose(row):

    name = ''

    directions = {0: 'n', 1: 's', 2: 'w', 3: 'e', 4: 'unknown'}

    for idx, value in enumerate(row):

        if value:

            name += directions[idx]

    return name
directions_train = move_directions(df_train)

directions_test = move_directions(df_test)



df_train['wind_rose'] = directions_train[['north', 'south', 'west', 'east', 'unknown']].apply(wind_rose, axis=1, raw=True)

df_test['wind_rose'] = directions_test[['north', 'south', 'west', 'east', 'unknown']].apply(wind_rose, axis=1, raw=True)
plt.figure(figsize=(20,2))

sns.heatmap(df_train.corr()[['fare_amount']].sort_values('fare_amount', ascending=False).iloc[1:].T, annot=True, cmap='viridis', vmax=0.88, vmin=-0.21)
fig = plt.figure(figsize=(10,8))



ax1 = fig.add_axes([0, 0, 1, 1])

ax2 = fig.add_axes([0.15, 0.5, 0.4, 0.4])

ax1.scatter(x='pickup_latitude', y='pickup_longitude', data=df_train.loc[0:10000], color='red', s=0.5)

ax2.scatter(x='pickup_latitude', y='pickup_longitude', data=df_train.loc[0:10000], color='blue', s=0.5)



ax1.set_xlabel('Latitude', fontsize=15)

ax1.set_ylabel('Longitude', fontsize=15)

ax1.set_title('Pickup Coordinates', fontsize=15)

ax2.set_xlabel('Latitude', fontsize=15)

ax2.set_ylabel('Longitude', fontsize=15)

ax2.set_title('Pickup Coordinates - Zoom', fontsize=15)



ax1.set_xlim((40.64, 40.825))

ax1.set_ylim((-74.05, -73.75))

ax2.set_xlim((40.725, 40.775))

ax2.set_ylim((-74.0, -73.95))
plt.figure(figsize=(12,6))

sns.distplot(df_train['fare_amount'], bins=80, kde=False)

sns.despine(top=True, bottom=True, left=True, right=True)
plot_data = df_train.copy()

plot_data['hour'] = plot_data['hour'].astype('int')



fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 18), sharex=True, sharey=True)

sns.set_palette('viridis')

axes = list(axes.ravel())

fig.delaxes(axes[-1])



y_min, y_max = (plot_data['fare_amount'].min()-2, plot_data['fare_amount'].max()+2)



for feature, ax  in zip(['passenger_count', 'year', 'month', 'day_of_month', 'day_of_week', 'hour', 'wind_rose'], axes):

    uniques = sorted(plot_data[feature].unique())

    for feature_value in uniques:

        my_df = plot_data[plot_data[feature] == feature_value]

        sns.regplot(x='distance_km', y='fare_amount', label=str(feature_value), data=my_df, ci=0,  ax=ax)        

        ax.set_title(feature)

        ax.set_xlim((-2, 30))

        ax.set_ylim((y_min, y_max))

        ncols = 2 if len(uniques) > 12 else 1

        ax.legend(title=str(feature), ncol=ncols, loc='best', bbox_to_anchor=(1.12, 1))

fig.tight_layout()
fig1, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20))



for feature, ax  in zip(['passenger_count', 'year', 'month', 'day_of_month', 'day_of_week', 'hour', 'wind_rose'], axes):

    uniques = sorted(plot_data[feature].unique())

    for feature_value in uniques:

        my_df = plot_data[plot_data[feature] == feature_value]        

        sns.barplot(x=feature, y='fare_amount', data=plot_data, ci=None, palette='viridis', ax=ax[0])

        sns.barplot(x=feature, y='distance_km', data=plot_data, ci=None, palette='viridis', ax=ax[1])

        sns.despine(top=True, bottom=True, left=True, right=True)        

fig1.tight_layout()
my_X = df_train[['distance_km', 'year']]

my_X = pd.get_dummies(my_X, columns=['year'])

my_y = df_train['fare_amount']



def step_features(df, base_feature, dummies):

    

    new_df = pd.DataFrame()

    for dummy_feature in dummies:

        new_df[base_feature + ' | ' + dummy_feature] = df[base_feature] * df[dummy_feature]

    

    return new_df



new_df = step_features(my_X, 'distance_km', my_X.drop('distance_km', axis=1).columns)

my_X = pd.concat([my_X.drop('distance_km', axis=1), new_df], axis=1)



lr = LinearRegression(fit_intercept=False).fit(my_X, my_y)



base_fee = lr.coef_[0:7]

price_per_km = lr.coef_[7:]



pd.DataFrame(data=[base_fee, price_per_km], columns=df_train['year'].unique(), index=['Base Fee', '$/km']).round(1)
df_train = pd.get_dummies(df_train, columns=['wind_rose'], prefix='', prefix_sep='')

df_test = pd.get_dummies(df_test, columns=['wind_rose'], prefix='', prefix_sep='')
features_on_off = {'key': False,

                   'fare_amount': False,

                   'pickup_datetime': False,

                   'pickup_longitude': False,

                   'pickup_latitude': False,

                   'dropoff_longitude': False,

                   'dropoff_latitude': False,

                   'passenger_count': True,

                   'day_of_week': True,

                   'day_of_month': True,

                   'month': True,

                   'year': True,

                   'hour': True,                   

                   'delta_latitude': True,

                   'delta_longitude': True,

                   'distance_km': True,

                   'n': True,

                   's': True,

                   'w': True,

                   'e': True,

                   'ne': True,

                   'nw': True,

                   'se': True,

                   'sw': True,

                   'unknown': True}
features_on = [key for key, status in features_on_off.items() if status]
X = df_train[features_on]

y = df_train['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = Pipeline([('std', StandardScaler()),

                #('pca', PCA()),

                ('classifier', RandomForestRegressor())])
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
t0 = time()

param_grid = {'classifier__n_estimators': [100],

              'classifier__max_depth': [10, 15, 20, 25],

              'classifier__min_samples_split': [2, 3, 4, 5],

              'classifier__min_samples_leaf': [1, 2, 3, 4, 5]

             }



grid_search = GridSearchCV(clf, param_grid, cv=2)

grid_search.fit(X_train, y_train)



print(f'Running time: {time()-t0:.2f}s')
grid_search.best_params_
grid_search.score(X_train, y_train)
y_pred = grid_search.predict(X_test)
pd.DataFrame({'Real Fare': y_test, 'Predicted Fare': y_pred}).describe().T.drop('count', axis=1)
pd.DataFrame({'Real Fare': y_test, 'Predicted Fare': y_pred}).head(10).T
mean_squared_error(y_test, y_pred).round()
df_test['n'], df_test['s'], df_test['w'], df_test['e'] = 0, 0, 0, 0

X = df_test[features_on]
y_pred = grid_search.predict(X)
submission = pd.DataFrame({'key': df_test['key'], 'fare_amount': y_pred})
submission.to_csv('submission.csv', index=False)