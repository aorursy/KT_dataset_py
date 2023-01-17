import pandas as pd

import datetime

import math as m
import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
pd.set_option('display.max_columns', 500)
train
train.isnull().sum()
len(train)
test.isnull().sum()
len(test)
test.columns
train.columns
train.nunique()
test.nunique()
train['test'] = 0

test['test'] = 1
all_t = pd.concat([train, test], axis=0, join='outer', join_axes=None, ignore_index=False,

          keys=None, levels=None, names=None, verify_integrity=False,

          copy=True)
all_t = all_t.reset_index()

all_t.drop('index',axis =1 , inplace= True)
all_t['day'] = pd.DatetimeIndex(all_t['start_time']).day
all_t['month'] = pd.DatetimeIndex(all_t['start_time']).month
all_t['seconds'] = ( pd.DatetimeIndex(all_t['start_time']).hour * 60) +  (pd.DatetimeIndex(all_t['start_time']).minute * 60)
all_t.drop('start_time', axis = 1 , inplace = True)
all_t.isnull().sum()
all_t.dist_VT
all_t['dist'] = all_t.dist.fillna(all_t.dist_VT)
def impute_column(dataset, column):

    dataset = dataset.copy()

    day_month = pd.DataFrame(dataset.groupby(['day','month'])[column].mean())

    month = pd.DataFrame(dataset.groupby('month')[column].mean())

    mean = dataset[column].mean()



    def firstNonNan(listfloats):

        for item in listfloats:

            try: 

                if m.isnan(item) == False:

                    return item

            except:

                i = 0



    def impute(row):

        if m.isnan(row[column]) == True:

            day_month_lat = day_month.loc[row[['day','month']]][column]

            month_lat = month.loc[row['month']][column]



            row[column] = firstNonNan([day_month_lat,month_lat, mean])

            assert(m.isnan(row[column]) == False)

        return row

    dataset = dataset.apply(impute, axis=1)

    print("{} imputed with mean".format(column))

    return dataset
for i in ['w_temp','w_visibility','w_windspeed','w_pressure','w_precipitation','w_humidity','w_dptemp']:

    print('i')

    all_t = impute_column(all_t,i)
all_t['log_distance'] = np.log(all_t['dist'] + 1)
def label_encoder(df):

    def numerical_features(df):

        columns = df.columns

        return df._get_numeric_data().columns



    def categorical_features(df):

        numerical_columns = numerical_features(df)

        return(list(set(df.columns) - set(numerical_columns)))

    

    categorical = categorical_features(df)

    # Creating the label encoder object

    le =  LabelEncoder()

    

    # Iterating over the "object" variables to transform the categories into numbers 

    for col in categorical:

        df[col] = le.fit_transform(df[col].astype(str))

    return df
all_t['speed'] = all_t[all_t['test'] == 0]['delivery_time']/all_t[all_t['test'] == 0]['dist']

speed_model = all_t[all_t['test'] == 0][['speed','van_model','delivery_time']].groupby('van_model').agg(['mean'])

all_t = all_t.join(speed_model, on ='van_model')

speed_package = all_t[all_t['test'] == 0][['speed','packages','delivery_time']].groupby('packages').agg(['mean'])

all_t = all_t.join(speed_package, on ='packages' , rsuffix = 'pack')
all_t['new_param'] = all_t['bearing'] * all_t['dist']
all_t_e = label_encoder(all_t)
import lightgbm as lgb
def rmsle(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))
X = np.array(all_t_e[all_t_e['test'] == 0].drop(['id', 'delivery_time','speed'], axis=1))

y = np.log(all_t_e[all_t_e['test'] == 0]['delivery_time'].values)

median_trip_duration = np.median(all_t_e[all_t_e['test'] == 0]['delivery_time'].values)



print('X.shape = ' + str(X.shape))

print('y.shape = ' + str(y.shape))



X_test = np.array(all_t_e[all_t_e['test'] == 1].drop(['id','speed','delivery_time'], axis=1))



print('X_test.shape = ' + str(X_test.shape))



print('Training and making predictions')

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmsle',

    'max_depth': 6, 

    'learning_rate': 0.1,

    'verbose': 0}

n_estimators = 600



n_iters = 5

preds_buf = []

err_buf = []

for i in range(n_iters): 

    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=i)

    d_train = lgb.Dataset(x_train, label=y_train)

    d_valid = lgb.Dataset(x_valid, label=y_valid)

    watchlist = [d_valid]



    model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=0  )



    preds = model.predict(x_valid)

    preds = np.exp(preds)

    preds[preds < 0] = median_trip_duration

    err = rmsle(np.exp(y_valid), preds)

    err_buf.append(err)

    print('RMSLE = ' + str(err))

    

    preds = model.predict(X_test)

    preds = np.exp(preds)

    preds[preds < 0] = median_trip_duration

    preds_buf.append(preds)



print('Mean RMSLE = ' + str(np.mean(err_buf)) + ' +/- ' + str(np.std(err_buf)))

# Average predictions

preds = np.mean(preds_buf, axis=0)
subm = pd.DataFrame()

subm['id'] = all_t[all_t['test']==1].id.values

subm['delivery_time'] = preds

subm.to_csv('submission_lgbm_600.csv', index=False)