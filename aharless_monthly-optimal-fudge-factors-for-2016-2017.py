LEARNING_RATE = 0.02            # shrinkage rate for boosting roudns

ROUNDS_PER_ETA = 20             # maximum number of boosting rounds times learning rate

VAL_SPLIT_DATE = '2016-09-15'   # First 2016 date not comparable to 2017 training data
import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

import datetime as dt

from datetime import datetime

import gc

import patsy

import statsmodels.api as smMonthly 

import statsmodels.formula.api as smf

from statsmodels.regression.quantile_regression import QuantReg
def calculate_aggregates(properties):

    # Number of properties in the zip

    zip_count = properties['regionidzip'].value_counts().to_dict()

    # Number of properties in the city

    city_count = properties['regionidcity'].value_counts().to_dict()

    # Median year of construction by neighborhood

    medyear = properties.groupby('regionidneighborhood')['yearbuilt'].aggregate('median').to_dict()

    # Mean square feet by neighborhood

    meanarea = properties.groupby('regionidneighborhood')['calculatedfinishedsquarefeet'].aggregate('mean').to_dict()

    # Neighborhood latitude and longitude

    medlat = properties.groupby('regionidneighborhood')['latitude'].aggregate('median').to_dict()

    medlong = properties.groupby('regionidneighborhood')['longitude'].aggregate('median').to_dict()

    return( zip_count, city_count, medyear, meanarea, medlat, medlong )

def munge(properties):

    for c in properties.columns:

        properties[c]=properties[c].fillna(-1)

        if properties[c].dtype == 'object':

            lbl = LabelEncoder()

            lbl.fit(list(properties[c].values))

            properties[c] = lbl.transform(list(properties[c].values))

def calculate_target_aggreagates(df):

    # Standard deviation of target value for properties in the city/zip/neighborhood

    citystd = df.groupby('regionidcity')['logerror'].aggregate("std").to_dict()

    zipstd = df.groupby('regionidzip')['logerror'].aggregate("std").to_dict()

    hoodstd = df.groupby('regionidneighborhood')['logerror'].aggregate("std").to_dict()

    return( citystd, zipstd, hoodstd )
def calculate_features(df):

    # Nikunj's features

    # Number of properties in the zip

    df['N-zip_count'] = df['regionidzip'].map(zip_count)

    # Number of properties in the city

    df['N-city_count'] = df['regionidcity'].map(city_count)

    # Does property have a garage, pool or hot tub and AC?

    df['N-GarPoolAC'] = ((df['garagecarcnt']>0) & \

                         (df['pooltypeid10']>0) & \

                         (df['airconditioningtypeid']!=5))*1 



    # More features

    # Mean square feet of neighborhood properties

    df['mean_area'] = df['regionidneighborhood'].map(meanarea)

    # Median year of construction of neighborhood properties

    df['med_year'] = df['regionidneighborhood'].map(medyear)

    # Neighborhood latitude and longitude

    df['med_lat'] = df['regionidneighborhood'].map(medlat)

    df['med_long'] = df['regionidneighborhood'].map(medlong)



    df['zip_std'] = df['regionidzip'].map(zipstd)

    df['city_std'] = df['regionidcity'].map(citystd)

    df['hood_std'] = df['regionidneighborhood'].map(hoodstd)
dropvars = ['parcelid', 'airconditioningtypeid', 'buildingclasstypeid',

            'buildingqualitytypeid', 'regionidcity']

droptrain = ['logerror', 'transactiondate']
xgb_params = {  # best as of 2017-09-28 13:20 UTC

    'eta': LEARNING_RATE,

    'max_depth': 7, 

    'subsample': 0.6,

    'objective': 'reg:linear',

    'eval_metric': 'mae',

    'lambda': 5.0,

    'alpha': 0.65,

    'colsample_bytree': 0.5,

    'silent': 1

}
num_boost_rounds = round( ROUNDS_PER_ETA / xgb_params['eta'] )

early_stopping_rounds = round( num_boost_rounds / 20 )

print('Boosting rounds: {}'.format(num_boost_rounds))

print('Early stoping rounds: {}'.format(early_stopping_rounds))
properties = pd.read_csv('../input/properties_2016.csv')

aggs = calculate_aggregates(properties)

zip_count, city_count, medyear, meanarea, medlat, medlong = aggs

munge(properties)

train = pd.read_csv("../input/train_2016_v2.csv")

train_df = train.merge(properties, how='left', on='parcelid')

del train

gc.collect()
for m in range(1,12+1):

    

    print( "\n\nFUDGE FACTOR ANALYSIS FOR 2016 MONTH ", m)

    select_mon = pd.to_datetime(train_df["transactiondate"]).dt.month==m



    select_data = select_mon & (pd.to_datetime(train_df["transactiondate"]) >= VAL_SPLIT_DATE)

    target_aggs = calculate_target_aggreagates(train_df[~select_data])

    citystd, zipstd, hoodstd = target_aggs



    train1 = train_df.copy()

    calculate_features(train1)



    x_valid = train1.drop(dropvars+droptrain, axis=1)[select_mon]

    y_valid = train1["logerror"].values.astype(np.float32)[select_mon]

    n_valid = x_valid.shape[0]



    train1=train1[ train1.logerror > -0.4 ]

    train1=train1[ train1.logerror < 0.419 ]



    train1=train1[~select_mon]



    # Use only training data comparable to what is available for 2017

    select_qtr4 = pd.to_datetime(train1["transactiondate"]) >= VAL_SPLIT_DATE

    train1=train1[~select_qtr4]

    

    x_train=train1.drop(dropvars+droptrain, axis=1)

    y_train = train1["logerror"].values.astype(np.float32)

    y_mean = np.mean(y_train)

    xgb_params['base_score'] = y_mean

    

    n_train = x_train.shape[0]



    print('Fitting model to {} points & using {} to fit fudge factor'.format(n_train, n_valid))



    dtrain = xgb.DMatrix(x_train, y_train)

    dvalid_x = xgb.DMatrix(x_valid)

    dvalid_xy = xgb.DMatrix(x_valid, y_valid)



    evals = [(dtrain,'train'),(dvalid_xy,'eval')]

    model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds,

                      evals=evals, early_stopping_rounds=early_stopping_rounds,

                      verbose_eval=False)



    valid_pred = model.predict(dvalid_x, ntree_limit=model.best_ntree_limit)

    fudge = QuantReg(y_valid, valid_pred).fit(q=.5).params[0]

    rawerr = mean_absolute_error(y_valid, valid_pred)

    fudgerr = mean_absolute_error(y_valid, fudge*valid_pred)

    print("Fudge factor reduces MAE from {0:9.6f} to {1:9.6f}".format(rawerr, fudgerr))

    print("Optimized fudge factor for month {0}: {1:9.6f}".format(m, fudge))

properties = pd.read_csv('../input/properties_2017.csv')

aggs = calculate_aggregates(properties)

zip_count, city_count, medyear, meanarea, medlat, medlong = aggs

munge(properties)

train = pd.read_csv("../input/train_2017.csv")

train_df = train.merge(properties, how='left', on='parcelid')

del train

gc.collect()
for m in range(1,9+1):

    

    print( "\n\nFUDGE FACTOR ANALYSIS FOR 2017 MONTH ", m)

    select_mon = pd.to_datetime(train_df["transactiondate"]).dt.month==m



    target_aggs = calculate_target_aggreagates(train_df[~select_mon])

    citystd, zipstd, hoodstd = target_aggs



    train1 = train_df.copy()

    calculate_features(train1)



    x_valid = train1.drop(dropvars+droptrain, axis=1)[select_mon]

    y_valid = train1["logerror"].values.astype(np.float32)[select_mon]

    n_valid = x_valid.shape[0]



    train1=train1[ train1.logerror > -0.4 ]

    train1=train1[ train1.logerror < 0.419 ]



    train1=train1[~select_mon]

    

    x_train=train1.drop(dropvars+droptrain, axis=1)

    y_train = train1["logerror"].values.astype(np.float32)

    y_mean = np.mean(y_train)

    xgb_params['base_score'] = y_mean

    

    n_train = x_train.shape[0]



    print('Fitting model to {} points & using {} to fit fudge factor'.format(n_train, n_valid))



    dtrain = xgb.DMatrix(x_train, y_train)

    dvalid_x = xgb.DMatrix(x_valid)

    dvalid_xy = xgb.DMatrix(x_valid, y_valid)



    evals = [(dtrain,'train'),(dvalid_xy,'eval')]

    model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds,

                      evals=evals, early_stopping_rounds=early_stopping_rounds,

                      verbose_eval=False)



    valid_pred = model.predict(dvalid_x, ntree_limit=model.best_ntree_limit)

    fudge = QuantReg(y_valid, valid_pred).fit(q=.5).params[0]

    rawerr = mean_absolute_error(y_valid, valid_pred)

    fudgerr = mean_absolute_error(y_valid, fudge*valid_pred)

    print("Fudge factor reduces MAE from {0:9.6f} to {1:9.6f}".format(rawerr, fudgerr))

    print("Optimized fudge factor for month {0}: {1:9.6f}".format(m, fudge))
