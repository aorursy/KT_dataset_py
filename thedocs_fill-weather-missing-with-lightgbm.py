import pandas as pd

import numpy as np 

import os 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import sklearn.metrics, sklearn.ensemble

from sklearn.svm import SVR

from lightgbm import LGBMRegressor

from pathlib import Path

data_dir = Path('../input/ashrae-energy-prediction')

# this is for train. Just switch below to paths to fill missing data in test weather

weather = pd.read_csv(data_dir/'weather_train.csv', parse_dates=['timestamp'], 

                      dtype={'site_id': np.uint16})

weather_test = pd.read_csv(data_dir/'weather_test.csv', parse_dates=['timestamp'], 

                      dtype={'site_id': np.uint16})



weather.head()

# running df that will be appended to 

running_batch = weather[weather['site_id'] == 1].set_index('timestamp').resample('h').mean().copy()

running_batch['site_id'] = 1

# for each site, resampling weather every one hour

for site in weather['site_id'].unique():

    if site == 1:

        continue



    site_batch = weather[weather['site_id'] == site].set_index('timestamp').resample('1h').mean()   

    site_batch['site_id'] = site

    running_batch = running_batch.append(site_batch)

print(running_batch.isna().sum())

print('Weather has increased by {} samples'.format(len(running_batch)-len(weather)))

    
weather = running_batch.reset_index(level=0).copy()

weather = weather.sort_values(['timestamp'])



weather['hour']=weather['timestamp'].apply(lambda x: x.hour).astype(np.uint8)

weather['month'] = weather['timestamp'].apply(lambda x: x.month).astype(np.uint8)

weather['day']=weather['timestamp'].apply(lambda x: x.day).astype(np.uint8)

weather['year']=(weather['timestamp'].apply(lambda x: x.year) - 2015).astype(np.uint8)





weather_test['hour']=weather_test['timestamp'].apply(lambda x: x.hour).astype(np.uint8)

weather_test['month'] = weather_test['timestamp'].apply(lambda x: x.month).astype(np.uint8)

weather_test['day']=weather_test['timestamp'].apply(lambda x: x.day).astype(np.uint8)

weather_test['year']=(weather_test['timestamp'].apply(lambda x: x.year) - 2015).astype(np.uint8)
corr = weather.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)
# label encode columns [list] of df 

def encode_df(df, columns):

        onehot_features = []

        for i, column in enumerate(columns):

            lenc = LabelEncoder()

            df[column] = lenc.fit_transform(df[column].values)

            

        return df

    

# get data from X columns test data 

def extract_x_from_test(df, cols):

    df = df[cols]

    df.dropna(inplace=True)

    return df 



# inputs: Dataframe, list, list, string. Fills rows in target column that are missing, by

# training on non missing values of target and features 

# features only include continous features

def fill_col(df, X_from_testset, features, target,  cat_features_names):    

    

    # updating feature set names 

    features += cat_features_names

    # onehot encoding cat columns 

    train_df  = df.copy()

    train_df = encode_df(train_df, cat_features_names)

    # extracting non-missing features and missing target rows for test 

    x_test =  train_df[(~train_df[features].isna()).all(axis=1)][train_df[target].isna()][features]

    # extracting non-missing features and target rows for train

    x_train =  train_df[(~train_df[features].isna()).all(axis=1)][~train_df[target].isna()]

    if len(X_from_testset) !=0:

        a = X_from_testset[features+[target]].dropna()

        x_train = x_train.append(a)

    

    # data from test set     

    # dataset specs 

    print('Training on {0:.5f} fraction, {1:} samples'.format(len(x_train)/len(train_df), len(x_train)))

    print('Filling up {0:.5f} fraction, {1:} samples'.format(len(x_test)/len(train_df),len(x_test)))

    

    if len(x_train) == 0 or len(x_test) == 0:

        print('Cannot fill any missing values.')

        return df



    y_train = x_train[target]

    x_train=x_train[features]



    # grid search cv

    param_grid = {'num_leaves': [15], 'learning_rate':[0.25],

                 'min_child_samples':[70], 'n_estimators':[45],

                  'lambda_l2':[20], 'max_bin':[50], 'objective':['regression']}

    

    gbm = LGBMRegressor(categorical_features=cat_features_names)

    gc = sklearn.model_selection.GridSearchCV(gbm, param_grid=param_grid, cv=4, verbose=1,

        n_jobs=6, refit='r2', scoring=['neg_mean_absolute_error', 'r2'], 

                                              return_train_score=True)

    # fits best scoring model 

    gc.fit(x_train, y_train)

    

    train_preds2 = gc.predict(X=x_train)

    test_preds = gc.predict(X=x_test)

    df.at[x_test.index, target] = test_preds

    

                  

    results = pd.DataFrame.from_dict(gc.cv_results_)

    results = results.sort_values(['rank_test_r2'])

    metrics=['mean_test_r2', 'mean_test_neg_mean_absolute_error', 'mean_train_r2', 'mean_train_neg_mean_absolute_error' ]

    eval_results = results.iloc[0][metrics]

    

    print(gc.best_params_)

    print(eval_results)



    return df



print('Missing values at start')

print(weather.isna().sum())
target = 'dew_temperature'

features = ['air_temperature', 'hour', 'month', 'year', 'day']

cat_features = ['site_id']



weather_f1 = fill_col(weather.copy(), weather_test.copy(), features, target, cat_features)  

target = 'cloud_coverage'

features=['dew_temperature', 'hour', 'month', 'wind_speed', 'year', 'day']

cat_features = ['site_id']

weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  

weather_f1.isna().sum()
target = 'precip_depth_1_hr'

features=['dew_temperature', 'hour', 'month', 'wind_speed', 'cloud_coverage',

          'year', 'day']



cat_features = ['site_id']

weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  

weather_f1.isna().sum()

target = 'sea_level_pressure'

features=['air_temperature', 'hour', 'month', 'wind_speed', 'cloud_coverage', 'precip_depth_1_hr', 

         'wind_direction', 'year', 'day']

cat_features = ['site_id']

weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  

weather_f1.isna().sum()

# predicting on due temperature again with missing values filled in

target = 'dew_temperature'

features=['hour', 'month', 'wind_speed', 'cloud_coverage', 'precip_depth_1_hr', 'year', 'day']

cat_features = ['site_id']

weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  

weather_f1.isna().sum()

target = 'wind_direction'

features=['hour', 'month', 'wind_speed', 'cloud_coverage', 

          'precip_depth_1_hr', 'dew_temperature', 'year', 'day']

cat_features = ['site_id']

weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  
weather_f1.isna().sum()
target = 'wind_direction'

features=['hour', 'month', 'year', 'day']

cat_features = ['site_id']

weather_f1 = fill_col(weather_f1.copy(), weather_test.copy(), features, target, cat_features)  

# getting df series with col names and if they cocntain missing values 

cols_ismissing = weather_f1.isna().any(axis=0).reset_index(level=0)

# getting missing column names 

missing_cols = cols_ismissing[cols_ismissing[0] == True]['index'].values

weather_f1 = weather_f1.sort_values(['timestamp'])

weather_f2 = weather_f1.copy()



for site_id in weather_f1['site_id'].unique():

    df = weather_f1[weather['site_id']==site_id].copy()

    if df.isna().any(axis=0).sum() == 0:

        continue

        

    weather_f2.at[df.index, missing_cols] = (df[missing_cols].fillna(method='bfill',limit =1) + 

                                             df[missing_cols].fillna(method='ffill', limit =1))/2

    

weather_f2.isna().sum()



previous_targets = []

for target in ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr',

                'sea_level_pressure', 'wind_direction','wind_speed']:

    print('Filling ', target)

    features=['hour', 'month', 'year'] + previous_targets

    cat_features = ['site_id', 'day'] 



    weather_f2 = fill_col(weather_f2.copy(), weather_test.copy(), features, target, cat_features)

    if target not in ['sea_level_pressure', 'wind_direction']:

        previous_targets.append(target)
weather_f2.isna().sum()
weather_f2.drop(['day', 'month', 'hour', 'year'], axis=1, inplace=True)

print(weather_f2.columns)

assert len(weather) == len(weather_f2)

weather_f2.to_csv('weather_train_filled.csv', index=False)