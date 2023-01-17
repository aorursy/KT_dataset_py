# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor, FeaturesData, Pool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import datetime

from datetime import timedelta

import os

print(os.listdir("../input"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
%time

# ,parse_dates=['booking_date', 'checkin_date', 'checkout_date']

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)

print(test.shape)
data = train.append(test,ignore_index = True) 

print(data.shape)

data.head()
%time

data.info()
for column in data:

    if data[column].dtype=='int64':

        data[column] = data[column].astype(np.int16)

for column in data:

    if data[column].dtype=='float64':

        data[column] = data[column].astype(np.float32)
data.info()
%time

data.describe()
%time

data.isna().sum()
# %time

# data.nunique()
print(data['booking_date'].max())

print(data['booking_date'].min())
data.head()
%time

def feature_engineering(df):

    

    df.loc[:,'booking_date'] = pd.to_datetime(df['booking_date'], format="%d/%m/%y",infer_datetime_format=True)

    df.loc[:,'checkin_date'] = pd.to_datetime(df['checkin_date'], format="%d/%m/%y",infer_datetime_format=True)

    df.loc[:,'checkout_date'] = pd.to_datetime(df['checkout_date'], format="%d/%m/%y",infer_datetime_format=True)

    

    df.loc[:,'checkin_day'] = df['checkin_date'].apply(lambda x : x.day)

    df.loc[:,'checkin_month'] = df['checkin_date'].apply(lambda x : x.month) 

    df.loc[:,'checkin_year'] = df['checkin_date'].apply(lambda x : x.year)

    df.loc[:,'checkin_day_of_year'] = df['checkin_date'].apply(lambda x : (x - datetime.datetime(x.year, 1, 1)).days + 1) 

    df.loc[:,'checkin_weekday'] = df['checkin_date'].apply(lambda x : x.weekday())    

    

    df.loc[:,'checkout_day'] = df['checkout_date'].apply(lambda x : x.day)

    df.loc[:,'checkout_month'] = df['checkout_date'].apply(lambda x : x.month) 

    df.loc[:,'checkout_year'] = df['checkout_date'].apply(lambda x : x.year)

    df.loc[:,'checkout_day_of_year'] = df['checkout_date'].apply(lambda x : (x - datetime.datetime(x.year, 1, 1)).days + 1) 

    df.loc[:,'checkout_weekday'] = df['checkout_date'].apply(lambda x : x.weekday()) 

    

    df.loc[:,'no_of_rooms'] = (df['numberofadults'] + df['numberofchildren'])/df['total_pax']

    df.loc[:,'trip_length'] = df['checkout_date'] - df['checkin_date']

    df.loc[:,'days_before_planning'] = df['checkin_date'] - df['booking_date']

    df.loc[:,'trip_length'] = df['trip_length'].apply(lambda x : x.days)

    df.loc[:,'days_before_planning'] = df['days_before_planning'].apply(lambda x : x.days)

    trip_count = df.groupby(['memberid'])['reservation_id'].agg(['count'])

    df.loc[:,'trip_count'] = df['memberid'].apply(lambda i: trip_count.loc[i][0])

    

    #handle data irregularity

    df['days_before_planning'] = df['days_before_planning'].apply(lambda x : x if x>=0 else 0)

    

    for column in df:

        if df[column].dtype=='int64':

            df[column] = df[column].astype(np.int16)

    for column in df:

        if df[column].dtype=='float64':

            df[column] = df[column].astype(np.float16)

    return df

    

# def drop(df):

#     to_drop = ['reservation_id','memberid','booking_date', 'checkin_date', 'checkout_date']

#     return df.drop(to_drop,axis=1)



def create_data(df):

    df = feature_engineering(df)

#     df = drop(df)

    return df
# trip_count = data.groupby(['memberid'])['reservation_id'].agg(['count'])

# trip_count = pd.DataFrame({'memberid':trip_count.index, 'trip_count':trip_count['count'].values})
# dataset['checkin_date'].head()
data.shape
# len(data['checkin_date'].apply(lambda x : x.weekday()))
%time

dataset = create_data(data)
# str((data.loc[0,'checkin_date'] + timedelta(days=k)).date())
dic = dict()

for i,r in dataset.iterrows():

    print(i)

#     ,r['resort_id'],r['checkin_date'].date(),r['checkout_date'].date(),r['trip_length']

    for k in range(r['trip_length']+1):

        if r['resort_id'] in dic.keys():

            if str((r['checkin_date'] + timedelta(days=k)).date()) in dic[r['resort_id']].keys():

                dic[r['resort_id']][str((r['checkin_date'] + timedelta(days=k)).date())] += 1

            else:

                dic[r['resort_id']][str((r['checkin_date'] + timedelta(days=k)).date())] = 1

        else:

            dic[r['resort_id']] = {}

            if str((dataset.loc[i,'checkin_date'] + timedelta(days=k)).date()) in dic[r['resort_id']].keys():

                dic[r['resort_id']][str((r['checkin_date'] + timedelta(days=k)).date())] += 1

            else:

                dic[r['resort_id']][str((r['checkin_date'] + timedelta(days=k)).date())] = 1

print(len(dic))
# dic['4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce']
maxdic = dict()

for key, value in dic.items():

    maxdic[key] = max(dic[key].values())

#     print(key, len(dic[key]))

print(maxdic)
def get_booking(row):

#     print(row)

#     print(row['resort_id'],str(row['checkin_date'].date()))

    return dic[row['resort_id']][str(row['checkin_date'].date())]
dataset.loc[:,'resort_max_bookings'] = dataset['resort_id'].map(maxdic) 
dataset.loc[:,'current_occupancy']=dataset.apply(lambda x : get_booking(x),axis=1)
# dataset.groupby(['resort_id','checkin_date'])['reservation_id'].agg(['count'])
dataset.columns
dataset.head()
# fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')

f, axs = plt.subplots(4,3,figsize=(20,20))

plt.subplot(4, 3, 1)

sns.countplot(dataset['channel_code'])

plt.xlabel('channel_code')

plt.ylabel('count')



plt.subplot(4, 3, 2)

sns.countplot(dataset['booking_type_code'])

plt.xlabel('booking_type_code')

plt.ylabel('count')



plt.subplot(4, 3, 3)

sns.countplot(dataset['cluster_code'])

plt.xlabel('cluster_code')

plt.ylabel('count')



plt.subplot(4, 3, 4)

sns.countplot(dataset['main_product_code'])

plt.xlabel('main_product_code')

plt.ylabel('count')



plt.subplot(4, 3, 5)

sns.countplot(dataset['member_age_buckets'])

plt.xlabel('member_age_buckets')

plt.ylabel('count')



plt.subplot(4, 3, 6)

sns.countplot(dataset['reservationstatusid_code'])

plt.xlabel('reservationstatusid_code')

plt.ylabel('count')



plt.subplot(4, 3, 7)

sns.countplot(dataset['resort_region_code'])

plt.xlabel('resort_region_code')

plt.ylabel('count')



plt.subplot(4, 3, 8)

sns.countplot(dataset['resort_type_code'])

plt.xlabel('resort_type_code')

plt.ylabel('count')



plt.subplot(4, 3, 9)

sns.countplot(dataset['room_type_booked_code'])

plt.xlabel('room_type_booked_code')

plt.ylabel('count')



plt.subplot(4, 3, 10)

sns.countplot(dataset['season_holidayed_code'])

plt.xlabel('season_holidayed_code')

plt.ylabel('count')



plt.subplot(4, 3, 11)

sns.countplot(dataset['state_code_residence'])

plt.xlabel('state_code_residence')

plt.ylabel('count')



plt.subplot(4, 3, 12)

sns.countplot(dataset['state_code_resort'])

plt.xlabel('state_code_resort')

plt.ylabel('count')







f, axs = plt.subplots(4,3,figsize=(20,20))

plt.subplot(3, 3, 1)

sns.distplot(train['amount_spent_per_room_night_scaled'],kde=False)

plt.xlabel('amount_spent_per_room_night_scaled')

# plt.ylabel('count')



plt.subplot(3, 3, 2)

sns.distplot(dataset['numberofadults'],kde=False)

plt.title('numberofadults')

# plt.ylabel('count')



plt.subplot(3, 3, 3)

sns.distplot(dataset['numberofchildren'],kde=False)

plt.title('cluster_code')

# plt.ylabel('count')



plt.subplot(3, 3, 4)

sns.distplot(dataset['roomnights'],kde=False)

plt.title('roomnights')

# plt.ylabel('count')



plt.subplot(3, 3, 5)

sns.distplot(dataset['trip_length'],kde=False)

plt.title('trip_length')

# plt.ylabel('count')



plt.subplot(3, 3, 6)

sns.distplot(dataset['total_pax'],kde=False)

plt.title('total_pax')

# plt.ylabel('count')





plt.subplot(3, 3, 7)

sns.distplot(dataset['trip_count'],kde=False)

plt.title('trip_count')



plt.subplot(3, 3, 8)

sns.distplot(dataset['days_before_planning'],kde=False)

plt.title('days_before_planning')
sns.distplot(train[train['season_holidayed_code']==1]['amount_spent_per_room_night_scaled'],label='1',color='r')

sns.distplot(train[train['season_holidayed_code']==2]['amount_spent_per_room_night_scaled'],label='2',color='y')

sns.distplot(train[train['season_holidayed_code']==3]['amount_spent_per_room_night_scaled'],label='3',color='g')

sns.distplot(train[train['season_holidayed_code']==4]['amount_spent_per_room_night_scaled'],label='4',color='b')

plt.xlabel('channel_code')

sns.distplot(train[train['room_type_booked_code']==1]['amount_spent_per_room_night_scaled'],label='1',color='r')

sns.distplot(train[train['room_type_booked_code']==2]['amount_spent_per_room_night_scaled'],label='2',color='y')

sns.distplot(train[train['room_type_booked_code']==3]['amount_spent_per_room_night_scaled'],label='3',color='g')

sns.distplot(train[train['room_type_booked_code']==4]['amount_spent_per_room_night_scaled'],label='4',color='b')

sns.distplot(train[train['room_type_booked_code']==5]['amount_spent_per_room_night_scaled'],label='4',color='b')

plt.xlabel('room_type_booked_code')
# dataset[dataset['days_before_planning']==0][['booking_date', 'checkin_date', 'checkout_date','roomnights','trip_length', 'days_before_planning']]
dataset.columns
cat =[]

for column in dataset:

    if 'id' in column or 'code' in column or 'check' in column:

        cat.append(column)

print(cat)
# Function to determine if column in dataframe is string.

def is_str(col):

    for i in col:

        if pd.isnull(i):

            continue

        elif isinstance(i, str):

            return True

        elif i in cat:

            return True

        else:

            return False

# Splits the mixed dataframe into categorical and numerical features.

def split_features(df):

    cfc = []

    nfc = []

    for column in df:

        if is_str(df[column]):

            cfc.append(column)

        else:

            nfc.append(column)

    return df[cfc], df[nfc]
def preprocess(cat_features, num_features):

    cat_features = cat_features.fillna("None")

    for column in num_features:

        num_features[column].fillna(np.nanmean(num_features[column]), inplace=True)

    return cat_features, num_features
y_train = dataset[dataset['amount_spent_per_room_night_scaled'].isnull()!=True]['amount_spent_per_room_night_scaled']

to_drop=['amount_spent_per_room_night_scaled','reservation_id','memberid','booking_date', 'checkin_date', 'checkout_date']

X_train = dataset[dataset['amount_spent_per_room_night_scaled'].isnull()!=True].drop(to_drop, axis=1)

X_test = dataset[dataset['amount_spent_per_room_night_scaled'].isnull()==True].drop(['reservation_id','memberid','booking_date', 'checkin_date', 'checkout_date'],axis=1)

# dftrain=dataset[dataset['amount_spent_per_room_night_scaled'].isnull()!=True]

# dftest=dataset[dataset['amount_spent_per_room_night_scaled'].isnull()==True]

# dftrain.head()
# Apply the "split_features" function on the data.

cat_tmp_train, num_tmp_train = split_features(X_train)

cat_tmp_test, num_tmp_test = split_features(X_test)
# Now to apply the "preprocess" function.

# Getting a "SettingWithCopyWarning" but I usually ignore it.

cat_features_train, num_features_train = preprocess(cat_tmp_train, num_tmp_train)

cat_features_test, num_features_test = preprocess(cat_tmp_test, num_tmp_test)
train_pool = Pool(

    data = FeaturesData(num_feature_data = np.array(num_features_train.values, dtype=np.float32), 

                    cat_feature_data = np.array(cat_features_train.values, dtype=object), 

                    num_feature_names = list(num_features_train.columns.values), 

                    cat_feature_names = list(cat_features_train.columns.values)),

    label =  np.array(y_train, dtype=np.float32)

)
test_pool = Pool(

    data = FeaturesData(num_feature_data = np.array(num_features_test.values, dtype=np.float32), 

                    cat_feature_data = np.array(cat_features_test.values, dtype=object), 

                    num_feature_names = list(num_features_test.columns.values), 

                    cat_feature_names = list(cat_features_test.columns.values))

)
CatBoostRegressor().get
model = CatBoostRegressor(iterations=4000,loss_function = 'RMSE', learning_rate=0.05, depth=7) 

# Fit model

model.fit(train_pool,early_stopping_rounds=2000)

# Get predictions

preds = model.predict(test_pool)
res_id = test['reservation_id']
df = pd.DataFrame({'reservation_id': res_id, 'amount_spent_per_room_night_scaled': preds}, columns=['reservation_id', 'amount_spent_per_room_night_scaled'])

df.to_csv("submission.csv", index=False)
# X,y=dftrain.drop('loan_default',axis=1),dftrain['loan_default']

# X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state = 1994)


# categorical_features_indices = np.where(X_train.dtypes =='object')[0]

# categorical_features_indices
# # import catboost



# class ModelOptimizer:

#     best_score = None

#     opt = None

    

#     def __init__(self, model, X_train, y_train, categorical_columns_indices=None, n_fold=3, seed=1994, early_stopping_rounds=30, is_stratified=True, is_shuffle=True):

#         self.model = model

#         self.X_train = X_train

#         self.y_train = y_train

#         self.categorical_columns_indices = categorical_columns_indices

#         self.n_fold = n_fold

#         self.seed = seed

#         self.early_stopping_rounds = early_stopping_rounds

#         self.is_stratified = is_stratified

#         self.is_shuffle = is_shuffle

        

        

#     def update_model(self, **kwargs):

#         for k, v in kwargs.items():

#             setattr(self.model, k, v)

            

#     def evaluate_model(self):

#         pass

    

#     def optimize(self, param_space, max_evals=10, n_random_starts=2):

#         start_time = time.time()

        

#         @use_named_args(param_space)

#         def _minimize(**params):

#             self.model.set_params(**params)

#             return self.evaluate_model()

        

#         opt = gp_minimize(_minimize, param_space, n_calls=max_evals, n_random_starts=n_random_starts, random_state=2405, n_jobs=-1)

#         best_values = opt.x

#         optimal_values = dict(zip([param.name for param in param_space], best_values))

#         best_score = opt.fun

#         self.best_score = best_score

#         self.opt = opt

        

#         print('optimal_parameters: {}\noptimal score: {}\noptimization time: {}'.format(optimal_values, best_score, time.time() - start_time))

#         print('updating model with optimal values')

#         self.update_model(**optimal_values)

#         plot_convergence(opt)

#         return optimal_values

    

# class CatboostOptimizer(ModelOptimizer):

#     def evaluate_model(self):

#         validation_scores = catboost.cv(

#         catboost.Pool(self.X_train, 

#                       self.y_train, 

#                       cat_features=self.categorical_columns_indices),

#         self.model.get_params(), 

#         nfold=self.n_fold,

#         stratified=self.is_stratified,

#         seed=self.seed,

#         early_stopping_rounds=self.early_stopping_rounds,

#         shuffle=self.is_shuffle,

#         verbose=100,

#         plot=False)

#         self.scores = validation_scores

#         test_scores = validation_scores.iloc[:, 2]

#         best_metric = test_scores.max()

#         return 1 - best_metric
# from skopt import gp_minimize

# from skopt.space import Real, Integer

# from skopt.utils import use_named_args

# from skopt.plots import plot_convergence

# import time
# cb = CatBoostRegressor(iterations=5000,loss_function = 'RMSE', learning_rate=0.05, depth=6,

#                          boosting_type='Ordered', # use permutations

#                          random_seed=1994, 

#                          use_best_model=True)

# # catboost.CatBoostClassifier(n_estimators=4000, # use large n_estimators deliberately to make use of the early stopping

# #                          loss_function='Logloss',

# #                          eval_metric='AUC',

# #                          boosting_type='Ordered', # use permutations

# #                          random_seed=1994, 

# #                          use_best_model=True)

# cb_optimizer = CatboostOptimizer(cb, X_train, y_train,categorical_columns_indices=categorical_features_indices)

# params_space = {'depth'         : [6,8,10],

#                   'learning_rate' : [0.01, 0.05, 0.1],

#                   'iterations'    : [3000, 5000, 10000]

#                  }

# cb_optimal_values = cb_optimizer.optimize(params_space)