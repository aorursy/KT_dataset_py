import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.inspection import plot_partial_dependence

import pdpbox

from pdpbox import pdp, get_dataset, info_plots

import xgboost as xgb

from sklearn.model_selection import ParameterGrid

from tqdm import tqdm



np.random.seed(1)

hour = pd.read_csv("../input/bike-sharing-dataset/hour.csv")
hour.head(6)
hour.shape
# Plottong data.
hour['season'].unique()
hour['yr'].unique()
hour['mnth'].unique()
hour['hr'].unique()
hour['holiday'].unique()
hour['weekday'].unique()
hour['workingday'].unique()
hour['weathersit'].unique()
plt.figure(figsize=(12, 6))

plt.plot(hour['temp'], '.')

plt.show()
plt.figure(figsize=(12, 6))

sns.distplot(hour['temp'])
plt.figure(figsize=(12, 6))

plt.plot(hour['atemp'], '.')

plt.show()
plt.figure(figsize=(12, 6))

sns.distplot(hour['atemp'])

# are they related?
plt.figure(figsize=(12, 6))

plt.plot(hour['hum'][:200], '.')

plt.show()
plt.figure(figsize=(12, 6))

sns.distplot(hour['hum'])
plt.figure(figsize=(12, 6))

plt.plot(hour['windspeed'], '.')

plt.show()
plt.figure(figsize=(12, 6))

sns.distplot(hour['windspeed'])
plt.figure(figsize=(12, 6))

plt.plot(hour['casual'], '.')

plt.show()
plt.figure(figsize=(12, 6))

sns.distplot(hour['casual'])
plt.figure(figsize=(12, 6))

plt.plot(hour['registered'], '.')

plt.show()
plt.figure(figsize=(12, 6))

sns.distplot(hour['registered'])
plt.figure(figsize=(12, 6))

plt.plot(hour['cnt'], '.')

plt.show()
plt.figure(figsize=(12, 6))

sns.distplot(hour['cnt'])
# Exclude casual and registered and use cnt as target.
# Number of days since the 01.01.2011 (the first day in the dataset). 

# This feature was introduced to take account of the trend over time.



hour['date'] = pd.to_datetime(hour['dteday'])



basedate = pd.Timestamp('2011-01-01')

hour['days_since'] = hour['date'].apply(lambda x: (x - basedate).days)
plt.figure(figsize=(12, 6))

plt.plot(hour['hum'], hour['days_since'], '.')

plt.show()
plt.figure(figsize=(12, 6))

sns.regplot(x=hour["temp"], y=hour["cnt"])
plt.figure(figsize=(12, 6))

sns.jointplot(x=hour["temp"], y=hour["cnt"], kind='scatter')
plt.figure(figsize=(12, 6))

sns.regplot(x=hour["atemp"], y=hour["cnt"])
plt.figure(figsize=(12, 6))

sns.regplot(x=hour["hum"], y=hour["cnt"])
plt.figure(figsize=(12, 6))

sns.regplot(x=hour["days_since"], y=hour["cnt"])
# For categorial features.

plt.figure(figsize=(12, 6))

sns.violinplot(x="season", y="cnt", data=hour, palette="muted")
plt.figure(figsize=(12, 6))

sns.boxplot(x="season", y="cnt", data=hour)
plt.figure(figsize=(12, 6))

sns.violinplot(x="yr", y="cnt", data=hour, palette="muted")
plt.figure(figsize=(12, 6))

sns.violinplot(x="mnth", y="cnt", data=hour, palette="muted")
plt.figure(figsize=(12, 6))

sns.violinplot(x="hr", y="cnt", data=hour, palette="muted")
plt.figure(figsize=(12, 6))

sns.boxplot(x="hr", y="cnt", data=hour)
plt.figure(figsize=(12, 6))

sns.violinplot(x="holiday", y="cnt", data=hour, palette="muted")
plt.figure(figsize=(12, 6))

sns.violinplot(x="weekday", y="cnt", data=hour, palette="muted")
plt.figure(figsize=(12, 6))

sns.violinplot(x="workingday", y="cnt", data=hour, palette="muted")
plt.figure(figsize=(12, 6))

sns.violinplot(x="weathersit", y="cnt", data=hour, palette="muted")
# Cyclical features. Hour, weekday and month. It is usefull for NN algorhitms. Tree algos are robust without it.



def encode_cyclical(data, col_name, max_val):

    data[col_name + '_sin'] = np.sin(2 * np.pi * data[col_name] / max_val)

    data[col_name + '_cos'] = np.cos(2 * np.pi * data[col_name] / max_val)

    return data





hour = encode_cyclical(hour, 'hr', 24)

hour = encode_cyclical(hour, 'mnth', 12)

hour = encode_cyclical(hour, 'weekday', 7)

features = ['season', 'yr', 'mnth', 'hr', 'holiday',

            'weekday', 'workingday', 'weathersit', 'temp', 'atemp',

            'hum', 'windspeed', 'days_since']



X, y = hour[features], hour['cnt']
def rand_forest_model(X, y):

    rmse_arr = []

    

    kf = KFold(n_splits=5, random_state=1, shuffle=True)



    for n, (train_index, val_index) in enumerate(kf.split(X, y)):

        print(f'fold: {n}')

        train_X = X.iloc[train_index].values

        val_X = X.iloc[val_index].values

        train_y = y[train_index].values

        val_y = y[val_index].values

        

        regr = RandomForestRegressor(max_depth=20, n_estimators=140, random_state=0)

        regr.fit(train_X, train_y)

        # print(regr.feature_importances_)



        y_pred = regr.predict(val_X)

        

        # Predicted values should be non negative.

        y_pred[y_pred < 0] = 0

        

        rmse = np.sqrt(mean_squared_error(val_y, y_pred))

        rmse_arr.append(rmse)

        

    print('RMSE list:', rmse_arr)

    print('RMSE AVG:', np.mean(rmse_arr))

    return {'rmse_arr': rmse_arr, 'y_pred': y_pred, 'y_val': val_y, 'train_X': train_X, 'model': regr}





res = rand_forest_model(X, y)
# Pot predicitons.

plt.figure(figsize=(12, 6))

plt.plot(res['y_pred'], '.', label='pred')

plt.plot(res['y_val'], '.', label='original')

plt.legend()

plt.show()

# plotting absolute deviation.

plt.figure(figsize=(12, 6))

plt.plot(np.abs(res['y_pred'] - res['y_val']), '.')

plt.title('Deviation from val.')

plt.show()

# Evaluating features importance for RFregressor model.

# res['model'].feature_importances_

plt.figure(figsize=(12, 6))

sns.barplot(x=res['model'].feature_importances_, y=features)

plt.title('Feature importances')
plot_partial_dependence(estimator=res['model'], X=res['train_X'], features=[(0, 2), 2], feature_names=features) 
plot_partial_dependence(estimator=res['model'], X=res['train_X'], features=[0, 1], feature_names=features) 
plot_partial_dependence(estimator=res['model'], X=res['train_X'], features=[2, 3], feature_names=features) 

plot_partial_dependence(estimator=res['model'], X=res['train_X'], features=[4, 5], feature_names=features) 
plot_partial_dependence(estimator=res['model'], X=res['train_X'], features=[6, 7], feature_names=features) 
plot_partial_dependence(estimator=res['model'], X=res['train_X'], features=[8, 9], feature_names=features) 
print(np.corrcoef(hour["temp"], hour["atemp"]))
plt.figure(figsize=(12, 6))

sns.regplot(x=hour["temp"], y=hour["atemp"])
plot_partial_dependence(estimator=res['model'], X=res['train_X'], features=[10, 11], feature_names=features) 
plot_partial_dependence(estimator=res['model'], X=res['train_X'], features=[12], feature_names=features) 
# Excluding holiday, "not very important" feature.

features = ['season', 'yr', 'mnth', 'hr',

            'weekday', 'workingday', 'weathersit', 'temp', 'atemp',

            'hum', 'windspeed', 'days_since']



X, y = hour[features], hour['cnt']



rand_forest_model(X, y)
# excluding atemp, "not very important" feature.

features = ['season', 'yr', 'mnth', 'hr', 'holiday',

            'weekday', 'workingday', 'weathersit', 'temp', 'atemp',

            'hum', 'windspeed', 'days_since']



X, y = hour[features], hour['cnt']



rand_forest_model(X, y)
# Excluding hour, "very important" feature.

features = ['season', 'yr', 'mnth', 'holiday',

            'weekday', 'workingday', 'weathersit', 'temp', 'atemp',

            'hum', 'windspeed', 'days_since']



X, y = hour[features], hour['cnt']



rand_forest_model(X, y)
# Lets try to remove days_since feature.

features = ['season', 'yr', 'mnth', 'hr', 'holiday',

            'weekday', 'workingday', 'weathersit', 'temp',

            'hum', 'windspeed']



X, y = hour[features], hour['cnt']



rand_forest_model(X, y)
features = ['season', 'yr', 'mnth', 'hr', 'holiday',

            'weekday', 'workingday', 'weathersit', 'temp',

            'hum', 'windspeed']



rmse_ft_arr = []



for n in range(len(features)):

    print(f'step {n}')

    features = ['season', 'yr', 'mnth', 'hr', 'holiday',

            'weekday', 'workingday', 'weathersit', 'temp',

            'hum', 'windspeed']

    features.remove(features[n])

    X, y = hour[features], hour['cnt']



    res1 = rand_forest_model(X, y)



    rmse_ft_arr.append(np.mean(res1['rmse_arr']))

features = ['season', 'yr', 'mnth', 'hr', 'holiday',

            'weekday', 'workingday', 'weathersit', 'temp',

            'hum', 'windspeed']



plt.figure(figsize=(12, 6))

sns.barplot(x=rmse_ft_arr, y=features)

plt.title('RMSE vs removed feature.')

features = ['season', 'yr', 'mnth', 'hr', 'holiday',

            'weekday', 'workingday', 'weathersit', 'temp', 'atemp',

            'hum', 'windspeed', 'days_since']



data_df = pd.DataFrame(res['train_X'], columns=features)



pdp_hr = pdp.pdp_isolate(

    model=res['model'], dataset=data_df, model_features=features, feature='hr', num_grid_points=200

)



fig, axes = pdp.pdp_plot(pdp_hr, 'hr', plot_lines=True, frac_to_plot=400)
plot_partial_dependence(estimator=res['model'], X=res['train_X'], features=[3], feature_names=features, grid_resolution=200) 
features = ['season', 'yr', 'mnth', 'hr', 'holiday',

            'weekday', 'workingday', 'weathersit', 'temp',

            'hum', 'windspeed', 'days_since']



X, y = hour[features], hour['cnt']



pars = {

    'learning_rate': 0.1,

    'max_depth': 12,

    'objective': 'reg:squarederror',

    'eval_metric': 'rmse',

    'gamma': 0.25,

    'n_estimators': 280

}



def xgb_model(X, y, pars):    

    rmse_arr = []

    

    kf = KFold(n_splits=5, random_state=1, shuffle=True)



    for n, (train_index, val_index) in enumerate(kf.split(X, y)):

#         print(f'fold: {n}')

        

        train_X = X.iloc[train_index]

        val_X = X.iloc[val_index]

        train_y = y[train_index]

        val_y = y[val_index]

        

        xgb_train = xgb.DMatrix(train_X, label=train_y)

        xgb_eval = xgb.DMatrix(val_X, label=val_y)

        

        xgb_model = xgb.train(pars,

              xgb_train,

              num_boost_round=800,

              evals=[(xgb_train, 'train'), (xgb_eval, 'val')],

              verbose_eval=False,

              early_stopping_rounds=30

             )

    

        y_pred = xgb_model.predict(xgb.DMatrix(val_X))



        rmse = np.sqrt(mean_squared_error(val_y, y_pred))

        rmse_arr.append(rmse)

        

    print('RMSE list:', rmse_arr)

    print('RMSE AVG:', np.mean(rmse_arr))

    return {'rmse_arr': rmse_arr, 'y_pred': y_pred, 'y_val': val_y, 'train_X': train_X, 'model': xgb_model}





features = ['season', 'yr', 'mnth', 'hr', 'holiday',

            'weekday', 'workingday', 'weathersit',

            'temp', 'hum', 'windspeed', 'days_since']



X, y = hour[features], hour['cnt']



res = xgb_model(X, y, pars)

par_grid = {

    "max_depth": [3, 6, 7, 8, 9],

    "min_child_weight": [0.5, 1, 3],

    "gamma": [0.25, 0.5, 0.8, 0.9, 1.1],

    "n_estimators": [60, 80, 100, 140]

#     "learning_rate": [0.05, 0.15, 0.25, 0.30],

#     "colsample_bytree": [0.3, 0.4, 0.5, 0.7, 0.9],

#     "etha": [0.01, 0.5, 0.1, 0.2],

#     "subsample": [0.5, 0.7, 1.0],

#     "lambda": [0.5, 1.0, 2.0]

}



rmse_avg_min = 1e10

min_pars = None





print('total:', len(ParameterGrid(par_grid)))

for n, par in enumerate(ParameterGrid(par_grid)):

    print(f'step {n}')

    

    model_pars = {

        'objective': 'reg:squarederror',

        'eval_metric': 'rmse',

    }

    

    for k, v in par.items():

        model_pars[k] = v 

    

    res = xgb_model(X, y, model_pars)

    

    rmse_avg = np.mean(res['rmse_arr'])

    

    if rmse_avg < rmse_avg_min:

        rmse_avg_min = rmse_avg

        min_pars = par



        

print(f'Best AVG RMSE: {rmse_avg_min}')

print('Best parameters:', min_pars)
print(f'With tunned parameters:', min_pars)

print('xgb model gives:')

res = xgb_model(X, y, min_pars)
plt.figure(figsize=(12, 6))

plt.plot(np.abs(res['y_pred'] - res['y_val']), '.')

plt.title('Abs deviation from validation set.')

plt.ylabel('Rentals')

plt.show()