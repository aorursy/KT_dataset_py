import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os

import warnings



%matplotlib inline

warnings.filterwarnings("ignore")
# Set the size of the plots 

plt.rcParams["figure.figsize"] = (18,8)

sns.set(rc={'figure.figsize':(18,8)})
data = pd.read_csv("../input/pubg-presentation-features-engineering/train.csv")

print("Finished loading the data")
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
data = reduce_mem_usage(data)
y = data['winPlacePerc']
X = data

X.drop(['Unnamed: 0', 'winPlacePerc'], inplace=True, axis=1)
from sklearn.metrics import mean_absolute_error as mae

from sklearn.model_selection import train_test_split

import lightgbm as lgb
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=12)
param = {

        'objective': 'regression',

        'metric': 'mae',

        'verbosity': -1,

        'lambda_l1': 1.16e-08,

        'lambda_l2': 0.18,

        'learning_rate': 0.1,

        'num_leaves': 206,

        'feature_fraction': 0.84,

        'bagging_fraction': 0.49,

        'bagging_freq': 2,

        'min_data_in_leaf': 39,

        'max_depth': 13,

        'early_stopping_round': 100

    }
n_rounds = 5000

d_train = lgb.Dataset(X_train, label=y_train)

d_valid = lgb.Dataset(X_test, label=y_test)

watchlist = [d_valid]
model = lgb.train(param, d_train, n_rounds, watchlist, verbose_eval=100)
pred = model.predict(X_test)

mae_scr = mae(y_test, pred)

print("SCORE:", mae_scr)
from sklearn.externals import joblib

# save model

joblib.dump(model, 'model_lgbm.pkl')