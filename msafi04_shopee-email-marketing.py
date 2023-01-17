import numpy as np

import pandas as pd

import os

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import gc



from sklearn.metrics import matthews_corrcoef

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display



for dirname, _, filenames in os.walk('/kaggle/input/shopee-code-league-20/_DA_Marketing_Analytics'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from time import time, strftime, gmtime



start = time()

print(start)



import datetime

print(str(datetime.datetime.now()))
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
train = pd.read_csv('/kaggle/input/shopee-code-league-20/_DA_Marketing_Analytics/train.csv')

print(train.shape)

train
test = pd.read_csv('/kaggle/input/shopee-code-league-20/_DA_Marketing_Analytics/test.csv')

print(test.shape)

test
users = pd.read_csv('/kaggle/input/shopee-code-league-20/_DA_Marketing_Analytics/users.csv')

print(users.shape)

users
train['grass_date'] = pd.to_datetime(train['grass_date']).dt.date

test['grass_date'] = pd.to_datetime(test['grass_date']).dt.date
train.describe().T
train.dtypes
users.describe().T
users.dtypes
plt.figure(figsize = (10, 10))

sns.countplot(train['open_flag'])
neg = train['open_flag'].value_counts().values[0]

pos = train['open_flag'].value_counts().values[1]

train['open_flag'].value_counts(normalize = True), pos, neg
plt.figure(figsize = (10, 10))

sns.distplot(users['age'])
plt.figure(figsize = (10, 10))

sns.boxplot(users['age'])
lbls, freqs = np.unique(train['country_code'].values, return_counts = True)

#print(list(zip(lbls, freqs)))



plt.figure(figsize = (10, 10))

plt.title('Train - Country Code')

plt.pie(freqs, labels = lbls, autopct = '%1.1f%%', shadow = False, startangle = 90)

plt.show()
lbls, freqs = np.unique(test['country_code'].values, return_counts = True)

#print(list(zip(lbls, freqs)))



plt.figure(figsize = (10, 10))

plt.title('Test - Country Code')

plt.pie(freqs, labels = lbls, autopct = '%1.1f%%', shadow = False, startangle = 90)

plt.show()
plt.figure(figsize = (10, 10))

train['grass_date'].hist()
print('***Checking Null values..***')

for col in train.columns:

    #print('****' * 10, col, '****' * 10)

    print('Train - ', col, ' : ', train[col].isnull().all())

print()

for col in test.columns:

    #print('****' * 10, col, '****' * 10)

    print('Test - ', col, ' : ', test[col].isnull().all())
for col in train.columns:

    print('Train - ', col, ' : ', pd.to_numeric(train[col], errors = 'coerce').notnull().all())

print()    

for col in test.columns:

    print('Test - ', col, ' : ', pd.to_numeric(train[col], errors = 'coerce').notnull().all())
#Another way to find the row index where the str appears

#train[~train.applymap(lambda x: isinstance(x, (int, float)))]
train['last_open_day'] = train['last_open_day'].replace('Never open', 0)

train['last_login_day'] = train['last_login_day'].replace('Never login', 0)

train['last_checkout_day'] = train['last_checkout_day'].replace('Never checkout', 0)



test['last_open_day'] = test['last_open_day'].replace('Never open', 0)

test['last_login_day'] = test['last_login_day'].replace('Never login', 0)

test['last_checkout_day'] = test['last_checkout_day'].replace('Never checkout', 0)
for col in ['last_open_day', 'last_login_day', 'last_checkout_day']:

    print('Train - ', col, ' : ', pd.to_numeric(train[col], errors = 'coerce').notnull().all())

print()    

for col in ['last_open_day', 'last_login_day', 'last_checkout_day']:

    print('Test - ', col, ' : ', pd.to_numeric(train[col], errors = 'coerce').notnull().all())
for col in users.columns:

    print('Users - ', col, ' : ', users[col].isnull().all())
for col in users.columns:

    print(users[col].value_counts(dropna = False))
users['attr_1'].fillna(2.0, inplace = True)

users['attr_2'].fillna(users['attr_2'].value_counts().index[0], inplace = True)

users['attr_3'].fillna(users['attr_3'].value_counts().index[0], inplace = True)

users['domain'].fillna(users['domain'].value_counts().index[0], inplace = True)
median = round(users['age'].median())

std = users['age'].std()

outliers = (users['age'] - median).abs() > std

users['age'][outliers] = np.nan

users['age'].fillna(median, inplace = True)
for col in users.columns:

    print(users[col].value_counts(dropna = False))
print(train.shape, test.shape)

train = pd.merge(train, users, on = 'user_id')

test = pd.merge(test, users, on = 'user_id')

print(train.shape, test.shape)

display(train.head(), test.head())
train['year'] = pd.to_datetime(train['grass_date']).dt.year

train['month'] = pd.to_datetime(train['grass_date']).dt.month

train['day'] = pd.to_datetime(train['grass_date']).dt.day



test['year'] = pd.to_datetime(test['grass_date']).dt.year

test['month'] = pd.to_datetime(test['grass_date']).dt.month

test['day'] = pd.to_datetime(test['grass_date']).dt.day



del train['grass_date'], test['grass_date'], train['user_id'], test['user_id'], train['row_id'], test['row_id']

gc.collect()
target = train['open_flag'].copy()

del train['open_flag']

gc.collect()
for col in train.columns:

    if train[col].dtype == 'object' and col != 'domain':

        train[col] = train[col].astype(np.int32)
for col in test.columns:

    if test[col].dtype == 'object' and col != 'domain':

        test[col] = test[col].astype(np.int32)
train.dtypes
cat_features = ['country_code', 'domain', 'year', 'month', 'day', 'attr_1', 'attr_2', 'attr_3']

num_features = [col for col in train.columns if col not in cat_features]

print(cat_features, num_features)
lbl = LabelEncoder()

for feature in cat_features:

    lbl.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = lbl.transform(list(train[feature].astype(str).values))

    test[feature] = lbl.transform(list(test[feature].astype(str).values))
%%time

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
Xtrain, Xvalid, ytrain, yvalid = train_test_split(train, target, test_size = 0.2, random_state = 42)

print(Xtrain.shape, ytrain.shape, Xvalid.shape, yvalid.shape)
import lightgbm as lgbm
pos_neg = np.sqrt(neg / pos)

pos_neg
params = {'num_leaves': 120,

          'min_child_weight': 0.001,

          'min_child_samples': 20,

          'feature_fraction': 0.379,

          'bagging_fraction': 0.8,

          'min_data_in_leaf': 50,

          'objective': 'binary',

          'max_depth': 10,

          'learning_rate': 0.002,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": {'auc'},

          "verbosity": -1,

          'reg_alpha': 0.389,

          'reg_lambda': 0.648,

          'scale_pos_weight': pos_neg,

          'random_state': 47,

         }
def lgb_mcc_score(y_pred, data):

    y_true = data.get_label()

    y_pred = np.round(y_pred)

    return 'mcc', matthews_corrcoef(y_true, y_pred), True



def lgb_mcc(preds, train_data):

    THRESHOLD = 0.5

    labels = train_data.get_label()

    return 'mcc', matthews_corrcoef(labels, preds >= THRESHOLD)
ltrain = lgbm.Dataset(Xtrain, label = ytrain, categorical_feature = cat_features)

lvalid = lgbm.Dataset(Xvalid, label = yvalid, categorical_feature = cat_features)



evals_result = {}



clf = lgbm.train(params, ltrain, 12000, valid_sets = [ltrain, lvalid], 

                 feval = lgb_mcc_score, evals_result = evals_result,

                 verbose_eval = 200, early_stopping_rounds = 1000)
lgbm.plot_metric(evals_result, metric = 'mcc')
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(), train.columns)), columns = ['Value','Features'])



plt.figure(figsize = (20, 10))

sns.barplot(x = "Value", y = "Features", data = feature_imp.sort_values(by = "Value", ascending = False))

plt.title('LightGBM Features)')

plt.tight_layout()

plt.show()

#plt.savefig('lgbm_importances-01.png')
predictions = clf.predict(test, verbose = 1)
predictions[:10]
sample_sub = pd.read_csv('/kaggle/input/shopee-code-league-20/_DA_Marketing_Analytics/sample_submission_0_1.csv')

sample_sub
sample_sub['open_flag'] = np.where(predictions > 0.5, 1, 0)

sample_sub.to_csv('./sample_sub_ShopeeEmail.csv', index = False)

sample_sub
sample_sub['open_flag'].value_counts(normalize = True)
plt.figure(figsize = (10, 10))

sns.countplot(sample_sub['open_flag'])
finish = time()

print(strftime("%H:%M:%S", gmtime(finish - start)))