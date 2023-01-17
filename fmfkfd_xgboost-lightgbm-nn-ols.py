import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

import gc

from sklearn.linear_model import LinearRegression

import random

import datetime as dt



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout, BatchNormalization

from keras.layers.advanced_activations import PReLU

from keras.layers.noise import GaussianDropout

from keras.optimizers import Adam

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

# Parameters

FUDGE_FACTOR = 1.1200  # Multiply forecasts by this



XGB_WEIGHT = 0.6200

BASELINE_WEIGHT = 0.0100

OLS_WEIGHT = 0.0620

NN_WEIGHT = 0.0800



XGB1_WEIGHT = 0.8000  # Weight of first in combination of two XGB models



BASELINE_PRED = 0.0115   # Baseline based on mean of training data, per Oleg
print( "\nReading data from disk ...")

prop = pd.read_csv('../input/properties_2016.csv')

train = pd.read_csv("../input/train_2016_v2.csv")
#LightGBM

print( "\nProcessing data for LightGBM ..." )

for c, dtype in zip(prop.columns, prop.dtypes):	

    if dtype == np.float64:

        prop[c] = prop[c].astype(np.float32)



df_train = train.merge(prop, how='left', on='parcelid')





missing_perc_thresh = 0.98

num_rows = df_train.shape[0]

for c in df_train.columns:

    #print (c)

    num_missing = df_train[c].isnull().sum()

    if num_missing == 0:

        continue

    missing_frac = num_missing / float(num_rows)

    if missing_frac > missing_perc_thresh:

        df_train.drop([c],axis=1)

        #exclude_missing.append(c)

#df_train=df_train.drop(exclude_missing,axis=1)

##39

##excluding having unique value



# exclude where we only have one unique value :D

for c in df_train.columns:

    num_uniques = len(df_train[c].unique())

    if df_train[c].isnull().sum() != 0:

        num_uniques -= 1

    if num_uniques == 1:

        df_train.drop([c],axis=1)

        #exclude_unique.append(c)

df_train.fillna(df_train.median(),inplace = True)



#x_train['Ratio_1'] = x_train['taxvaluedollarcnt']/x_train['taxamount']

                         

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc','propertycountylandusecode', 'fireplacecnt', 'fireplaceflag'], axis=1)



y_train = df_train['logerror'].values

print(x_train.shape, y_train.shape)





train_columns = x_train.columns



for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)



del df_train; gc.collect()



x_train = x_train.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)



params = {}

params['max_bin'] = 10

params['learning_rate'] = 0.0021 # shrinkage_rate

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'l1'          # or 'mae'

params['sub_feature'] = 0.345    # feature_fraction (small values => use very different submodels)

params['bagging_fraction'] = 0.85 # sub_row

params['bagging_freq'] = 40

params['num_leaves'] = 512        # num_leaf

params['min_data'] = 500         # min_data_in_leaf

params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

params['verbose'] = 0

params['feature_fraction_seed'] = 2

params['bagging_seed'] = 3



np.random.seed(0)

random.seed(0)

print("\nFitting LightGBM model ...")

clf = lgb.train(params, d_train, 430)



del d_train; gc.collect()

del x_train; gc.collect()



print("\nPrepare for LightGBM prediction ...")

print("   Read sample file ...")

sample = pd.read_csv('../input/sample_submission.csv')

print("   ...")

sample['parcelid'] = sample['ParcelId']

print("   Merge with property data ...")

df_test = sample.merge(prop, on='parcelid', how='left')

print("   ...")

del sample, prop; gc.collect()

print("   ...")

#df_test['Ratio_1'] = df_test['taxvaluedollarcnt']/df_test['taxamount']

x_test = df_test[train_columns]

print("   ...")

del df_test; gc.collect()

print("   Preparing x_test...")

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)

print("   ...")

x_test = x_test.values.astype(np.float32, copy=False)



print("\nStart LightGBM prediction ...")

p_test = clf.predict(x_test)



del x_test; gc.collect()



print( "\nUnadjusted LightGBM predictions:" )

print( pd.DataFrame(p_test).head() )



#XGBOOST



print( "\nRe-reading properties file ...")

properties = pd.read_csv('../input/properties_2016.csv')



print( "\nProcessing data for XGBoost ...")

for c in properties.columns:

    properties[c]=properties[c].fillna(-1)

    if properties[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(properties[c].values))

        properties[c] = lbl.transform(list(properties[c].values))



train_df = train.merge(properties, how='left', on='parcelid')

missing_perc_thresh = 0.98

num_rows = train_df.shape[0]

for c in train_df.columns:

    #print (c)

    num_missing = train_df[c].isnull().sum()

    if num_missing == 0:

        continue

    missing_frac = num_missing / float(num_rows)

    if missing_frac > missing_perc_thresh:

        train_df.drop([c],axis=1)

        #exclude_missing.append(c)

#df_train=df_train.drop(exclude_missing,axis=1)

##39

##excluding having unique value



# exclude where we only have one unique value :D

for c in train_df.columns:

    num_uniques = len(train_df[c].unique())

    if train_df[c].isnull().sum() != 0:

        num_uniques -= 1

    if num_uniques == 1:

        train_df.drop([c],axis=1)

x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)

x_test = properties.drop(['parcelid'], axis=1)

# shape        

print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))
#outliers

train_df=train_df[ train_df.logerror > -0.4 ]

train_df=train_df[ train_df.logerror < 0.419 ]

x_train=train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)

y_train = train_df["logerror"].values.astype(np.float32)

y_mean = np.mean(y_train)



print('After removing outliers:')     

print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))



print("\nSetting up data for XGBoost ...")

# xgboost params

xgb_params = {

    'eta': 0.037,

    'max_depth': 5,

    'subsample': 0.80,

    'objective': 'reg:linear',

    'eval_metric': 'mae',

    'lambda': 0.8,   

    'alpha': 0.4, 

    'base_score': y_mean,

    'silent': 1

}



dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)



num_boost_rounds = 250

print("num_boost_rounds="+str(num_boost_rounds))



# train model

print( "\nTraining XGBoost ...")

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)



print( "\nPredicting with XGBoost ...")

xgb_pred1 = model.predict(dtest)



print( "\nFirst XGBoost predictions:" )

print( pd.DataFrame(xgb_pred1).head() )

print("\nSetting up data for XGBoost ...")

# xgboost params

xgb_params = {

    'eta': 0.033,

    'max_depth': 6,

    'subsample': 0.80,

    'objective': 'reg:linear',

    'eval_metric': 'mae',

    'base_score': y_mean,

    'silent': 1

}



num_boost_rounds = 150

print("num_boost_rounds="+str(num_boost_rounds))



print( "\nTraining XGBoost again ...")

model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)



print( "\nPredicting with XGBoost again ...")

xgb_pred2 = model.predict(dtest)



print( "\nSecond XGBoost predictions:" )

print( pd.DataFrame(xgb_pred2).head() )

xgb_pred = XGB1_WEIGHT*xgb_pred1 + (1-XGB1_WEIGHT)*xgb_pred2

#xgb_pred = xgb_pred1



print( "\nCombined XGBoost predictions:" )

print( pd.DataFrame(xgb_pred).head() )
del train_df

del x_train

del x_test

del properties

del dtest

del dtrain

del xgb_pred1

del xgb_pred2 

gc.collect()
#OLS



np.random.seed(17)

random.seed(17)



print( "\n\nProcessing data for OLS ...")



train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

properties = pd.read_csv("../input/properties_2016.csv")

submission = pd.read_csv("../input/sample_submission.csv")

print(len(train),len(properties),len(submission))
def get_features(df):

    df["transactiondate"] = pd.to_datetime(df["transactiondate"])

    df["transactiondate_year"] = df["transactiondate"].dt.year

    df["transactiondate_month"] = df["transactiondate"].dt.month

    df['transactiondate'] = df['transactiondate'].dt.quarter

    df = df.fillna(-1.0)

    return df



def MAE(y, ypred):

    #logerror=log(Zestimate)−log(SalePrice)

    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / len(y)
train = pd.merge(train, properties, how='left', on='parcelid')

y = train['logerror'].values

test = pd.merge(submission, properties, how='left', left_on='ParcelId', right_on='parcelid')

properties = [] #memory



exc = [train.columns[c] for c in range(len(train.columns)) if train.dtypes[c] == 'O'] + ['logerror','parcelid']

col = [c for c in train.columns if c not in exc]



train = get_features(train[col])

test['transactiondate'] = '2016-01-01' #should use the most common training date

test = get_features(test[col])
print("\nFitting OLS...")

reg = LinearRegression(n_jobs=-1)

reg.fit(train, y); print('fit...')

print(MAE(y, reg.predict(train)))

train = [];  y = [] #memory



test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']

test_columns = ['201610','201611','201612','201710','201711','201712']

#combining all predictions



print( "\nCombining XGBoost, LightGBM, NN, and baseline predicitons ..." )

#lgb_weight = 1 - XGB_WEIGHT - BASELINE_WEIGHT - NN_WEIGHT - OLS_WEIGHT 

lgb_weight = 1 - XGB_WEIGHT - BASELINE_WEIGHT - OLS_WEIGHT 

lgb_weight0 = lgb_weight / (1 - OLS_WEIGHT)

xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)

baseline_weight0 =  BASELINE_WEIGHT / (1 - OLS_WEIGHT)

#nn_weight0 = NN_WEIGHT / (1 - OLS_WEIGHT)

pred0 = 0

pred0 += xgb_weight0*xgb_pred

pred0 += baseline_weight0*BASELINE_PRED

pred0 += lgb_weight0*p_test

#pred0 += nn_weight0*nn_pred

print( "\nCombined XGB/LGB/NN/baseline predictions:" )

print( pd.DataFrame(pred0).head() )



print( "\nPredicting with OLS and combining with XGB/LGB/NN/baseline predicitons: ..." )

for i in range(len(test_dates)):

    test['transactiondate'] = test_dates[i]

    pred = FUDGE_FACTOR * ( OLS_WEIGHT*reg.predict(get_features(test)) + (1-OLS_WEIGHT)*pred0 )

    submission[test_columns[i]] = [float(format(x, '.4f')) for x in pred]

    print('predict...', i)



print( "\nCombined XGB/LGB/NN/baseline/OLS predictions:" )

print( submission.head() )
##### WRITE THE RESULTS



from datetime import datetime



print( "\nWriting results to disk ..." )

submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)



print( "\nFinished ...")