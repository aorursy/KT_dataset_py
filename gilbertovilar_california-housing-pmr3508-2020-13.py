import os

import sys

import pandas as pd

import numpy as np

import warnings

import seaborn as sns

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

plt.style.use('seaborn-darkgrid')

plt.rc('xtick',labelsize=15)

plt.rc('ytick',labelsize=15)

parameters = {

    'axes.titlesize': 25,

    'axes.labelsize': 20,

    'legend.fontsize': 20,

    'grid.alpha': 0.6

}

plt.rcParams.update(parameters)
train = pd.read_csv("../input/atividade-regressao-PMR3508/train.csv")

train
train.describe()
train.info()
train.isna().sum()
train.drop(columns=['Id'], inplace=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,16))

train['longitude'].plot(kind='hist', bins=train['longitude'].value_counts().count(), ax=ax1, alpha=0.8)

train.plot(kind='scatter', x='longitude', y='median_house_value', color='orange', alpha=0.3, ax=ax2)

ax1.set_title('Histogram')

ax1.set_xlabel('Longitude')

ax2.set_title('Preço x Longitude')

print(train[['median_house_value', 'longitude']].corr())

plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,16))

train['latitude'].plot(kind='hist', bins=train['latitude'].value_counts().count(), ax=ax1, alpha=0.8)

train.plot(kind='scatter', x='latitude', y='median_house_value', color='orange', alpha=0.3, ax=ax2)

ax1.set_title('Histogram')

ax1.set_xlabel('latitude')

ax2.set_title('Preço x latitude')

print(train[['median_house_value', 'latitude']].corr())

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(16,8))

sns.scatterplot(data=train,x="longitude",y="latitude",hue="median_house_value", ax=ax);
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,16))

train['median_age'].plot(kind='hist', bins=train['median_age'].value_counts().count(), ax=ax1, alpha=0.8)

train.plot(kind='scatter', x='median_age', y='median_house_value', color='orange', alpha=0.3, ax=ax2)

ax1.set_title('Histogram')

ax1.set_xlabel('median_age')

ax2.set_title('Preço x median_age')

print(train[['median_house_value', 'median_age']].corr())

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18,16))

train.plot(kind='scatter', x='total_rooms', y='median_house_value', color='orange', alpha=0.3, ax=ax1)

train.plot(kind='scatter', x='total_bedrooms', y='median_house_value', color='orange', alpha=0.3, ax=ax2)

train.plot(kind='scatter', x='population', y='median_house_value', color='orange', alpha=0.3, ax=ax3)

train.plot(kind='scatter', x='households', y='median_house_value', color='orange', alpha=0.3, ax=ax4)



# ax1.set_title('Histogram')

# ax1.set_xlabel('total_rooms')

# ax2.set_title('Preço x total_rooms')

# print(train[['median_house_value', 'total_rooms']].corr())

sns.pairplot(train, vars=["total_rooms","total_bedrooms","population","households", "median_house_value"])

plt.show()
metrics = train[['total_rooms', 'total_bedrooms', 'population', 'households', 'median_house_value']]

metrics['rooms_per_household'] = metrics['total_rooms']/metrics['households']

metrics['bedrooms_per_household'] = metrics['total_bedrooms']/metrics['households']

metrics['otherrooms_per_household'] = (metrics['total_rooms'] - metrics['total_bedrooms'])/metrics['households']

metrics['people_per_bedroom'] = metrics['population']/metrics['total_bedrooms']

metrics['people_per_house'] = metrics['population']/metrics['households']
sns.pairplot(metrics, vars=['rooms_per_household', 

                            'bedrooms_per_household',

                            'otherrooms_per_household',

                            'people_per_bedroom',

                            'people_per_house',

                            'median_house_value'])

plt.show()
metrics['rooms_per_household'] = np.minimum(20, metrics['total_rooms']/metrics['households'])

metrics['bedrooms_per_household'] = np.minimum(5, metrics['total_bedrooms']/metrics['households'])

metrics['otherrooms_per_household'] = np.minimum(15, (metrics['total_rooms'] - metrics['total_bedrooms'])/metrics['households'])

metrics['people_per_bedroom'] = np.minimum(25, metrics['population']/metrics['total_bedrooms'])

metrics['people_per_house'] = np.minimum(25, metrics['population']/metrics['households'])
sns.pairplot(metrics, vars=['rooms_per_household', 

                            'bedrooms_per_household',

                            'otherrooms_per_household',

                            'people_per_bedroom',

                            'people_per_house',

                            'median_house_value'])

plt.tight_layout()

plt.show()
metrics.corr()['median_house_value'].to_frame()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,16))

train['median_income'].plot(kind='hist', bins=train['median_income'].value_counts().count(), ax=ax1, alpha=0.8)

train.plot(kind='scatter', x='median_income', y='median_house_value', color='orange', alpha=0.3, ax=ax2)

ax1.set_title('Histogram')

ax1.set_xlabel('median_income')

ax2.set_title('Preço x median_income')

print(train[['median_house_value', 'median_income']].corr())

plt.show()
plt.figure(figsize=(12, 6))

sns.heatmap(data=train.corr(), cmap='hot', linewidths=0.3, annot=True)
X = train.drop(columns = [

    'latitude',

    'longitude',

    'total_rooms',

    'total_bedrooms',

    'population',

    'households',

    'median_age',

    'median_house_value'

])

y = train['median_house_value']
from geopy.distance import geodesic



LA = (34.0522, -118.2437)

SF = (37.7749, -122.4194)



def city_distance(row):

    pair = (row['latitude'], row['longitude'])

    dist = min(geodesic(pair, SF).km, geodesic(pair, LA).km)

    return dist



X['city_distance'] = train.apply(city_distance, axis=1) 
X['rooms_per_household'] = np.minimum(20, train['total_rooms']/train['households'])

X['bedrooms_per_household'] = np.minimum(5, train['total_bedrooms']/train['households'])

X['otherrooms_per_household'] = np.minimum(15, (train['total_rooms'] - train['total_bedrooms'])/train['households'])

X['people_per_bedroom'] = np.minimum(25, train['population']/train['total_bedrooms'])

X['people_per_house'] = np.minimum(25, train['population']/train['households'])

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X = scaler.fit_transform(X)
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split





def rmsle(y_true, y_pred):

    return np.sqrt(np.mean((np.log(np.abs(y_pred)+1) - np.log(np.abs(y_true)+1))**2))



RMSLE_scorer = make_scorer(rmsle, greater_is_better=False)



def build_regressor(X_train, y_train, X_test, y_test, regressor, score, param_grid, n_folds=5):

    print("# Tuning hyper-parameters for %s" % score)

    print()

    reg = GridSearchCV(regressor, param_grid, cv=n_folds, scoring=score, verbose=True, n_jobs=3, iid=False)

    reg.fit(X_train, y_train)

    print("Best parameters set found cross-validation:")

    print()

    print(reg.best_params_)

    means = reg.cv_results_['mean_test_score']

    stds = reg.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, reg.cv_results_['params']):

        print("%0.3f (+/-%0.03f) for %r" %(mean, std * 2, params))

    y_true, y_pred = y_test, reg.predict(X_test)

    cross_val_score = np.min(np.abs(means))

    test_r2 = r2_score(y_true, y_pred)

    test_RMSLE = rmsle(y_true, y_pred)

    print()

    print("Cross validation and test results:")

    print()

    print(f'cross_val_score = {cross_val_score}')

    print(f'test_r2 = {test_r2}')

    print(f'test_RMSLE = {test_RMSLE}')

    return reg, cross_val_score, test_r2, test_RMSLE
from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



R = Ridge()

R_params = {

    'alpha': list(np.linspace(0.01, 10, 50)),

    'fit_intercept': [True, False],

    'normalize': [True, False]

}

L = Lasso()

L_params = {

    'alpha': list(np.linspace(0.01, 10, 50)),

    'fit_intercept': [True, False],

    'normalize': [True, False]

}

RF = RandomForestRegressor()

RF_params = {

    'n_estimators': [100, 200, 500],

    'max_depth': [5, 10, 20]

}

MLP = MLPRegressor()

MLP_params = {

    'hidden_layer_sizes': [(100, 100,), (100, 100, 100,), (100, 100, 100, 100, 100, 100,)],

    'max_iter': [200, 300]

}



regressors = [

    {

        'name': 'Ridge',

        'reg': R,

        'params': R_params

    },

    {

        'name': 'Lasso',

        'reg': L,

        'params': L_params

    },

    {

        'name': 'RandomForestRegressor',

        'reg': RF,

        'params': RF_params

    },

    {

        'name': 'MLPRegressor',

        'reg': MLP,

        'params': MLP_params

    }

]

score = RMSLE_scorer

results = []

for regressor in regressors:

    print(f'Starting Cross Validation for {regressor["name"]}')

    try:

        reg, cross_val_score, test_r2, test_RMSLE = build_regressor(X_train, y_train, X_test, y_test, 

                                                                    regressor['reg'], RMSLE_scorer, 

                                                                    regressor['params'])

    except:

        print(f"Couldnt grisearch for {regressor['name']}")

    results.append([regressor["name"], reg, cross_val_score, test_r2, test_RMSLE])

results_df = pd.DataFrame(data=results, columns=['name', 'regressor', 'cross_val_score', 'test_r2', 'test_RMSLE'])
results_df
min_RMSLE = results_df['test_RMSLE'].min()

best_model = results_df[results_df['test_RMSLE']==min_RMSLE]['name'].values[0]
print(best_model)

best_model_params = results_df[results_df['name']==best_model]['regressor'].values[0].best_params_

print(best_model_params)
best_model = RandomForestRegressor(**best_model_params)

best_model.fit(X, y)
test = pd.read_csv('../input/atividade-regressao-PMR3508/test.csv')

test.info()
X_test = test.drop(columns = [

    'Id',

    'latitude',

    'longitude',

    'total_rooms',

    'total_bedrooms',

    'population',

    'households',

    'median_age',

])



X_test['city_distance'] = test.apply(city_distance, axis=1)

X_test['rooms_per_household'] = np.minimum(20, test['total_rooms']/test['households'])

X_test['bedrooms_per_household'] = np.minimum(5, test['total_bedrooms']/test['households'])

X_test['otherrooms_per_household'] = np.minimum(15, (test['total_rooms'] - test['total_bedrooms'])/test['households'])

X_test['people_per_bedroom'] = np.minimum(25, test['population']/test['total_bedrooms'])

X_test['people_per_house'] = np.minimum(25, test['population']/test['households'])



X_test = scaler.transform(X_test)
y_pred = best_model.predict(X_test)
submission = pd.DataFrame()

submission['Id'] = test.Id

submission['median_house_value'] = y_pred

submission.to_csv('submission.csv', index=False)