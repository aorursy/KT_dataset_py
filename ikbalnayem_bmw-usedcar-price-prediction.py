import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/bmw-pricing-challenge/bmw_pricing_challenge.csv')

df.head()
y = df['price']

x = df.drop(['price', 'maker_key'], axis=1)
sns.jointplot(x=x['engine_power'], y=y, kind='reg', height=10)
for feature in 'feature_1 feature_2 feature_3 feature_4 feature_5 feature_6 feature_7 feature_8'.split():

    x[feature] = pd.get_dummies(x[feature], drop_first=True)
plt.figure(figsize=(10,10))

corr = pd.concat([x,y], axis=1).corr()

sns.heatmap(corr, annot=True)
plt.figure(figsize=(20,10))

sns.boxplot(x=x['model_key'], y=y)
model_key = {key: i for i, key in enumerate(x['model_key'].unique())}

x['model_key'] = x['model_key'].map(model_key)
fuel = {key: i for i, key in enumerate(x['fuel'].unique())}

x['fuel'] = x['fuel'].map(fuel)
x.head()
plt.figure(figsize=(18,5))

plt.subplot(1,2,1)

sns.boxplot(x['car_type'], y)

plt.subplot(1,2,2)

sns.boxplot(x['paint_color'], y)
x.drop(['car_type', 'paint_color'], axis=1, inplace=True)
x.head()
plt.figure(figsize=(20,10))

sns.boxplot(x['sold_at'], y)
x['registration_date'] = pd.to_datetime(x['registration_date'])

x['registration_year'] = x['registration_date'].dt.year

x['registration_month'] = x['registration_date'].dt.month

x.drop(['registration_date', 'sold_at'], axis=1, inplace=True)
x.head()
from sklearn.model_selection import train_test_split, RandomizedSearchCV



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)
params = {

    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],

    'n_estimators': [1, 5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 500, 1000, 2000],

    'max_leaf_nodes': [5, 10, 15, 20, 30, 40, 50, 55, 60, 70, 80, 85, 90, 95, 100],

    'random_state': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 35, 40]

}
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



model = RandomForestRegressor()

randomizedCV = RandomizedSearchCV(model, param_distributions=params, cv=5, verbose=3)

randomizedCV.fit(x, y)
randomizedCV.best_estimator_
model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=15,

                      max_features='auto', max_leaf_nodes=70,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, n_estimators=1000,

                      n_jobs=None, oob_score=False, random_state=7, verbose=0,

                      warm_start=False)

model.fit(x, y)

model.score(x_test, y_test)
mean_absolute_error(model.predict(x_test), y_test)