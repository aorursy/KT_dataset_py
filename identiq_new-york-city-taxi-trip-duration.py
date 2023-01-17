# Load libraries on Python 3 environment 



import pandas as pd

import numpy as np

import seaborn as sb 

import matplotlib.pyplot as plt

import matplotlib.dates as dates

from PIL import Image

import requests

from io import StringIO

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from scipy import stats

from datetime import datetime

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



%matplotlib inline
# Lookup given features on train and test data



# LOCAL_PATH = '~/Downloads/nyc-taxi-trip-duration'

KAGGLE_PATH = '../input'

data_train = pd.read_csv(KAGGLE_PATH + '/train.csv')

data_test = pd.read_csv(KAGGLE_PATH + '/test.csv')

sample_sub = pd.read_csv(KAGGLE_PATH + '/sample_submission.csv')



PLAN_URL = 'http://taxomita.com/wp-content/uploads/2017/12/map-of-areas-in-nyc-highway-map-of-new-york-city-metropolitan-area-highways.gif';

print(data_train.info())

print(data_test.info())

print('We have {} training rows and {} test rows.'.format(data_train.shape[0], data_test.shape[0]))

print('We have {} training columns and {} test columns.'.format(data_train.shape[1], data_test.shape[1]))

data_train.head(3)
"""

Extract date, year, month, weekday, hour from columns

"""

def datetime_extract(df, columns, modeling=False):

    df_ = df.copy()

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for col in columns:

        try:

            prefix = col

            if "_" in col:

                prefix = col.split("_")[0]

            ts = f"{prefix}_ts"

            df_[ts] = pd.to_datetime(df_[col])

            df_[f"{prefix}_month"] = df_[ts].dt.month

            df_[f"{prefix}_weekday"] = df_[ts].dt.weekday

            df_[f"{prefix}_day"] = df_[ts].dt.day

            df_[f"{prefix}_hour"] = df_[ts].dt.hour

            df_[f"{prefix}_minute"] = df_[ts].dt.minute

            if not modeling: 

                df_[f"{prefix}_date"] = df_[ts].dt.date

                df_[f"{prefix}_dayname"] = df_[f"{prefix}_weekday"].apply(lambda x: day_names[x])

            else:

                df_.drop(columns=[ts, col], axis = 1)

        except:

            pass

    return df_



"""

Extract delta between two timestamps in minutes

"""

def timedelta_extract(df, colname, start, end):

    df_= df.copy()

    df_[f'{colname}'] = (df_[end] - df_[start]).astype('timedelta64[m]')

    return df_



df_train = datetime_extract(data_train, ['pickup_datetime', 'dropoff_datetime'])

df_train = timedelta_extract(df_train, 'delta_m', 'pickup_ts', 'dropoff_ts')
df_train.head(3)
df_train.vendor_id.value_counts()
fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(14, 5))

for i, col in enumerate(['pickup', 'dropoff']):

    ax[i].plot(df_train.groupby(f'{col}_date').count()['id'], 'o-')

    ax[i].set(xlabel='Months', ylabel=f'{col} count'.title(), title=f'{col}s per date'.title())

plt.show()
fig, ax = plt.subplots(ncols=2, figsize=(14, 5))

for i, col in enumerate(['pickup', 'dropoff']):

    ax[i].plot(df_train.groupby([f'{col}_date']).sum()['passenger_count'])

    ax[i].set(xlabel='Months', ylabel="Total passengers", title="Total passengers per date")

    sb.distplot(df_train.passenger_count, kde=False, bins=df_train.passenger_count.max(), 

                vertical=True, axlabel="Passengers distribution");

    df_train.passenger_count.value_counts(sort=False)

# Download not working on Kaggle.. Try the PLAN_URL directly at the top of the page 

try:

    plan = requests.get(PLAN_URL)

    img = Image.open(BytesIO(plan.content))

    fig, ax = plt.subplots(figsize=(14, 12))

    ax.imshow(np.asarray(img), aspect='auto')

except:

    pass
# Some pickups / dropoffs are outside NYC area, we are dropping outliers (geopoint > 95% and < 5%)

# https://www.kaggle.com/misfyre/in-depth-nyc-taxi-eda-also-w-animation



def rm_geo_outliers(df, columns):

    df_ = df.copy()

    for i, col in enumerate(columns):

        col_lat = f"{col}_latitude"

        col_lng = f"{col}_longitude"

        df_ = df_[(

             df_[col_lng]>df_[col_lng].quantile(0.005))

           &(df_[col_lng]<df_[col_lng].quantile(0.995))

           &(df_[col_lat]>df_[col_lat].quantile(0.005))                           

           &(df_[col_lat]<df_[col_lat].quantile(0.995))]

    return df_



def display_geo(df, columns):

    for i, col in enumerate(columns):

        col_lat = f"{col}_latitude"

        col_lng = f"{col}_longitude"

        sb.lmplot(x=col_lng, y=col_lat, fit_reg=False, height=9, scatter_kws={'alpha':0.3,'s':5},

                       data=df)



        plt.xlabel(f'{col} Longitude'.title());

        plt.ylabel(f'{col} Latitude'.title());

        plt.show()

    return



geo_columns = ['pickup', 'dropoff'];

df_train = rm_geo_outliers(df_train, geo_columns)

display_geo(df_train, geo_columns)

df_train.store_and_fwd_flag.value_counts()
vendor_counts = df_train['vendor_id'].value_counts()



sb.barplot(vendor_counts.index, vendor_counts.values)

plt.xlabel('vendor_id')

plt.ylabel('Total rides')

plt.show()
df_train.nlargest(5, 'trip_duration')[['id', 'trip_duration', 'delta_m']]
# We have some outliers, let's remove them (> .97 quantile)



fig, ax = plt.subplots(figsize=(14, 4))

tripduration = df_train[df_train.trip_duration < df_train.trip_duration.quantile(.97)]

tripduration.groupby('delta_m').count()['id'].plot()



plt.xlabel('Trip duration in minutes')

plt.ylabel('Trip count')

plt.title('Duration distribution')

plt.show()
fig, ax = plt.subplots(figsize=(14, 4))

pd.pivot_table(tripduration, index='pickup_hour' ,aggfunc=np.mean)['trip_duration'].plot(label='mean')



plt.legend(loc=0)

plt.xlabel('Pickup hours (24h)')

plt.ylabel('Rides')

plt.title('Rides vs. pickup hours')

plt.show()
df_train = datetime_extract(data_train, ['pickup_datetime', 'dropoff_datetime'], modeling=True)

df_test = datetime_extract(data_test, ['pickup_datetime'], modeling=True)
def label_encode(df, column):

    df_ = df.copy();

    try:

        le = LabelEncoder()

        le.fit(data_train[column])

        df_train[column] = le.transform(df_train[column])

        df_test[column] = le.transform(df_test[column])

    except:

        pass

    return df_



df_train = label_encode(df_train, 'store_and_fwd_flag')

df_test = label_encode(df_test, 'store_and_fwd_flag')
predict_ids = df_test['id']
s_train, s_test = train_test_split(df_train, test_size = 0.2)

# Locally : s_train, s_test = train_test_split(df_train[0:100000], test_size = 0.2)
DROP_TRAIN = ['id', 'pickup_datetime', 'pickup_ts', 'dropoff_datetime', 'dropoff_ts']

DROP_PREDICT = DROP_TRAIN + ['trip_duration', 'dropoff_month', 'dropoff_weekday', 'dropoff_day',  'dropoff_hour', 'dropoff_minute'];
X_train = s_train.drop(DROP_PREDICT + ['trip_duration'], axis = 1)

Y_train = s_train["trip_duration"]

Y_train = Y_train.reset_index().drop('index', axis = 1)



X_test = s_test.drop(DROP_PREDICT + ['trip_duration'], axis = 1)

Y_test = s_test["trip_duration"]

Y_test = Y_test.reset_index().drop('index', axis = 1)
dtrain = xgb.DMatrix(X_train, label=np.log(Y_train+1))
dvalid = xgb.DMatrix(X_test, label=np.log(Y_test+1))
dtest = xgb.DMatrix(df_test.drop(['id', 'pickup_datetime', 'pickup_ts'], axis = 1))
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
 # https://www.kaggle.com/karelrv/nyct-from-a-to-z-with-xgboost-tutorial

"""

md = [6]

lr = [0.1,0.3]

mcw = [20,25,30]

for d in md:

    for l in lr:

        for w in mcw:

            t0 = datetime.now()

            

            xgb_pars = {'min_child_weight': w, 'eta': l, 'colsample_bytree': 0.9, 

                        'max_depth': d,

            'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}

            print('min_child_weight: {} | eta: {} | max-depth: {}.'.format(w, l, d))



            model = xgb.train(xgb_pars, dtrain, 50, watchlist, early_stopping_rounds=10,

                  maximize=False, verbose_eval=1)

  """
nrounds = 200

params = {'min_child_weight': 20, 'eta': 0.3, 'colsample_bytree': 0.9, 

            'max_depth': 6,

            'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}

model = xgb.train(params, dtrain, nrounds, watchlist, early_stopping_rounds=2,

      maximize=False, verbose_eval=1)

print('Modeling RMSLE %.5f' % model.best_score)

xgb.plot_importance(model, max_num_features=14, height=.5)
#n_folds = 5

#early_stopping = 10

#cv = xgb.cv(params, dtrain, 500, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=1)

pred = np.exp(model.predict(dtest)) - 1
df_pred = pd.DataFrame({'id': predict_ids, 'trip_duration': pred})

# Locally : df_pred = pd.DataFrame({'id': predict_ids[:20000], 'trip_duration': pred}) 

df_pred = df_pred.set_index('id')

df_pred.to_csv('submission.csv', index = True)
