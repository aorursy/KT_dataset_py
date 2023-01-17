import datetime as dt

import pandas as pd

import numpy as np 

import pandas_profiling as pf

from sklearn import model_selection

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')



#To suppress scientific notation

pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.set_option('display.max_columns', 300)
## Path

path_train = "../input/dsajunho4vs/train_df.csv"

path_test  = "../input/dsajunho4vs/test_df.csv"



## Read



### Train

dtype_categ = {'card_id'             : 'category',

               'feature_1'           : 'category',

               'feature_2'           : 'category',

               'feature_3'           : 'category',

               'first_active_year'   : 'category',

               'first_active_months' : 'category',

               'is_na'               : 'category'}



train_df = pd.read_csv(path_train, encoding="utf-8", dtype=dtype_categ)

#train_df = pd.read_csv(path_train, encoding="utf-8", dtype=dtype_categ, nrows = 1000)



### Test

test_df = pd.read_csv(path_test, encoding="utf-8", dtype=dtype_categ)

#test_df = pd.read_csv(path_test, encoding="utf-8", dtype=dtype_categ, nrows = 1000)
print("Rows and columns - Train: ",train_df.shape)

print("Rows and columns - Test: ",test_df.shape)
train_df.info()

train_df.describe(include='all')
train_df.head()
train_df.isnull().sum()
test_df.info()

test_df.describe(include='all')
test_df.head()
test_df.isnull().sum()
def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))
# Train

train_df['new_city_id'] = train_df['new_city_id'].fillna(-1)

train_df['aut_city_id'] = train_df['aut_city_id'].fillna(-1)

cols = train_df.select_dtypes([np.number]).columns

train_df[cols] = train_df[cols].fillna(-999)

train_df[pd.isnull(train_df).any(axis=1)].head()
# Test

test_df['new_city_id'] = test_df['new_city_id'].fillna(-1)

test_df['aut_city_id'] = test_df['aut_city_id'].fillna(-1)

cols = test_df.select_dtypes([np.number]).columns

test_df[cols] = test_df[cols].fillna(-999)

test_df[pd.isnull(test_df).any(axis=1)].head()
# Train

target = train_df['target']

del train_df['target']



train_df.head(3)
seed = 12345

validation_size = 0.20



train_x, valid_x, train_y, valid_y = train_test_split(train_df, 

                                                      target, 

                                                      test_size=validation_size,

                                                      random_state=seed)



print("Shape Train X: ", train_x.shape)

print("Shape Valid X: ", valid_x.shape)
iterations = 1000



params = {'iterations': iterations,

          'eval_metric': 'RMSE',

          'verbose': True}
from catboost import CatBoostRegressor

from catboost import cv, Pool



num_folds = 5



cat_features = ['card_id',

                'feature_1',

                'feature_2',

                'feature_3',

                'first_active_year',

                'first_active_months',

                'is_na',

                'new_city_id',

                'aut_city_id']



cv_dataset = Pool(data=train_x,

                  label=train_y,

                  cat_features=cat_features)



cv_results = cv(cv_dataset,

                params,

                fold_count=num_folds,

                plot=True,

                verbose=True)
model = CatBoostRegressor(iterations=iterations,

                          eval_metric='RMSE',

                          random_seed = seed)



model.fit(train_x, 

          train_y,

          cat_features=cat_features,

          verbose=False)
model.get_best_score()

print(model.get_params())
feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,valid_x)), columns=['Value','Feature'])



plt.figure(figsize=(10, 50))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('CatBoost Features')

plt.tight_layout()

plt.show()
valid_pred = model.predict(valid_x)

valid_pred[1:5]
print(rmse(valid_y, valid_pred))
model = CatBoostRegressor(iterations=iterations,

                          eval_metric='RMSE',

                          random_seed = seed)



model.fit(train_df, 

          target,

          cat_features=cat_features,

          verbose=False)



test_pred = model.predict(test_df)

test_pred[1:5]
sub_df = pd.DataFrame({'card_id': test_df['card_id'].values})

sub_df['target'] = test_pred

sub_df.to_csv('submit_catboost_1m_4vs_fm.csv', index=False)