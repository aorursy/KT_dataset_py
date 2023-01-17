import math

import os



import numpy as np

import pandas as pd

import geopy.distance



import catboost as cb



dataset_path = '/kaggle/input/covid19-global-forecasting-week-1/'





print ('catboost version', cb.__version__)
task_type = 'CPU'
def get_hubei_coords(df):

    for index, row in df.iterrows():

        if row['Province/State'] == 'Hubei':

            return (row['Lat'], row['Long'])



    raise Exception('Hubei not found in data')





def preprocess(df, hubei_coords, first_date):

    df.fillna({'Province/State': ''}, inplace=True)

    

    df['Day'] = (df['Date'] - first_date).dt.days.astype('int32')



    hubei_coords = get_hubei_coords(df)

    

    distance_to_hubei = []

    week_day = []

        

    for index, row in df.iterrows():

        distance_to_hubei.append(geopy.distance.distance((row['Lat'], row['Long']), hubei_coords).km)

        week_day.append(row['Date'].weekday())



    df['Distance_to_Hubei'] = distance_to_hubei

    df['WeekDay'] = week_day

    

    return df



df = pd.read_csv(os.path.join(dataset_path, 'train.csv'), parse_dates=['Date'])

df.drop(columns=['Id'], inplace=True)



test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'), parse_dates=['Date'])



hubei_coords = get_hubei_coords(df)

first_date = min(df['Date'])





df = preprocess(df, hubei_coords, first_date)

test_df = preprocess(test_df, hubei_coords, first_date)



print ('df.head', df.head())
last_train_date = pd.Timestamp(2020,3,11)

train_df = df[df['Date'] <= last_train_date].copy()
train_df.sort_values(by=['Date'])
prediction_types = ['ConfirmedCases', 'Fatalities']





train_labels = dict([(prediction_type, np.log1p(train_df[prediction_type])) for prediction_type in prediction_types])

train_df.drop(columns=['ConfirmedCases', 'Fatalities'], inplace=True)
train_df.drop(columns=['Date'], inplace=True)

test_df.drop(columns=['Date'], inplace=True)
cat_features = ['Province/State', 'Country/Region', 'WeekDay']





grid = {

    'learning_rate': [0.05, 0.07, 0.09, 0.3],

    'depth': [5, 6, 7],

    'l2_leaf_reg': [1, 3, 5, 7, 9],

    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']

}



submissions = {'ForecastId': test_df['ForecastId'].values}



for prediction_type in prediction_types:

    print ('prediction_type %s' % prediction_type)

    

    train_pool = cb.Pool(train_df, label=train_labels[prediction_type], cat_features=cat_features)

    test_pool = cb.Pool(test_df, cat_features=cat_features)

    

    model = cb.CatBoostRegressor(

        task_type=task_type,

        loss_function='RMSE',   # RMSE with log1p-transformed labels is RMSLE

        early_stopping_rounds=100,

        has_time=True,

        iterations=5000

    )



    model.randomized_search(grid, X=train_pool)

    

    feature_importance = model.get_feature_importance(prettified=True)

    print ('feature_importance', feature_importance)

    

    # truncate negative predictions to 0 and round

    # transform back from log1p using exp1m

    # round to integers

    submissions[prediction_type] = np.round(np.expm1(np.maximum(model.predict(test_pool), 0.0)))
submissions_df = pd.DataFrame(submissions)

print ('submissions_df.head()', submissions_df.head())



submissions_df.to_csv('submission.csv', index=False)