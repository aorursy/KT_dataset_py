import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn import model_selection

import warnings 



import matplotlib.pyplot as plt

import seaborn as sns



pd.options.mode.chained_assignment = None

warnings.filterwarnings('ignore')



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.isnull().sum(), '\n' ,test.isnull().sum(), '\n')

print(train.info(), '\n' ,test.info(), '\n')
y_train = train.price

x_train = train.drop(['id', 'price'], axis=1)

x_test = test.drop(['id'], axis=1)



index = x_train.shape[0]

df = pd.concat([x_train, x_test], axis=0)

    

year = df.date.apply(lambda x: x[0:4]).astype(int)

month = df.date.apply(lambda x: x[4:6]).astype(int)

day = df.date.apply(lambda x: x[6:8]).astype(int)

    

df['year_month'] = year*100 + month

df['month_day'] = month*100 + day

df['ym_freq'] = df.groupby('year_month')['year_month'].transform('count')

df['md_freq'] = df.groupby('month_day')['month_day'].transform('count')

df = df.drop(['date'], axis=1)



x_train = df.iloc[:index, :]

x_test = df.iloc[index:, :]



x_train.shape, x_test.shape



fig, ax = plt.subplots(6, 4, figsize=(20, 30))

n = 0

cols = x_train.columns

for r in range(6):

    for c in range(4):

        sns.kdeplot(x_train[cols[n]], ax=ax[r][c])

        ax[r][c].set_title(cols[n], fontsize=20)

        n += 1

        if n == x_train.shape[1]:

            break
def preprocessing(df):

    

    df.sqft_living = np.log(df.sqft_living)

    df.sqft_lot = np.log(df.sqft_lot)

    df.sqft_above = np.log(df.sqft_above)

    df.sqft_basement = np.log(df.sqft_basement)

    df.sqft_lot15 = np.log(df.sqft_lot15)

    

    df['roomsum'] = np.log(df.bedrooms + df.bathrooms)

    df['roomsize'] = df.sqft_living / df.roomsum

    

    df['pos'] = df.long.astype(str) + ', ' + df.lat.astype(str)

    df['density'] = df.groupby('pos')['pos'].transform('count')

    

    df = df.drop(['pos'], axis=1)

    

    return df



x_train = preprocessing(x_train)

x_test = preprocessing(x_test)



fig, ax = plt.subplots(5, 5, figsize=(20, 30))

n = 0

cols = x_train.columns

for r in range(5):

    for c in range(5):

        sns.kdeplot(x_train[cols[n]], ax=ax[r][c])

        ax[r][c].set_title(cols[n], fontsize=20)

        n += 1

        if n == x_train.shape[1]:

            break
xgb_params = {

    'eta': 0.01,

    'max_depth': 6,

    'subsample': 0.8,

    'colsample_bytree': 0.8,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



print('Transform DMatrix...')

dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)



print('Start Cross Validation...')



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=20,verbose_eval=50, show_stdv=False)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

print('best num_boost_rounds = ', len(cv_output))

rounds = len(cv_output)
model = xgb.train(xgb_params, dtrain, num_boost_round = rounds)

preds = model.predict(dtest)



sub = test[['id']]

sub['price'] = preds

sub.to_csv('sub_xgb_starter.csv', index=False)