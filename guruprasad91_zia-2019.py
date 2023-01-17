# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

pd.set_option('display.max_columns', 160)

import numpy as np
train = pd.read_csv("/kaggle/input/loan_train.csv")

test = pd.read_csv("/kaggle/input/new_loan_test.csv")
def handleNullValues(df):

    return df.loc[:, df.isnull().mean() < .75]
cols = ["DAYS_BIRTH","DAYS_EMPLOYED","CODE_GENDER","NAME_EDUCATION_TYPE","NAME_INCOME_TYPE","AMT_INCOME_TOTAL","DAYS_REGISTRATION","DAYS_ID_PUBLISH","FLAG_OWN_REALTY","FLAG_OWN_CAR","DAYS_LAST_PHONE_CHANGE"]

x = train[["SK_ID_CURR","DAYS_BIRTH","DAYS_EMPLOYED","CODE_GENDER","NAME_EDUCATION_TYPE","NAME_INCOME_TYPE","AMT_INCOME_TOTAL","DAYS_REGISTRATION","DAYS_ID_PUBLISH","FLAG_OWN_REALTY","FLAG_OWN_CAR","DAYS_LAST_PHONE_CHANGE"]]

y = train["TARGET"]

x=x.infer_objects() 

y=y.infer_objects()

test1 = test[x.columns]

test1 = test1.infer_objects()
x.dtypes
test1.dtypes
cols = x.select_dtypes("object").columns

cols



d = defaultdict(PandasLabelEncoder)

print(cols)

for a in cols:

    encoding = x.groupby(a).size()

    # get frequency of each category

    encoding = encoding/len(x)

    x[a] = x[a].map(encoding)

    test1[a] = test1[a].map(encoding)

x = x.fillna(0)

from sklearn.model_selection import train_test_split

x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
import lightgbm as lgb





def run_lgb(X_train, Y_train, X_valid, Y_valid, test):

    params = {

        "objective" : "regression",

        "metric" : "rmse",

        "task": "train",

        "boosting type":'dart',

        "num_leaves" :100,

        "learning_rate" : 0.01,

        "bagging_fraction" : 0.8,

        "feature_fraction" : 0.8,

        "bagging_frequency" : 6,

        "bagging_seed" : 42,

        "verbosity" : -1,

        "seed": 42

    }

    

    lgtrain = lgb.Dataset(X_train, label=Y_train)

    lgval = lgb.Dataset(X_valid, label=Y_valid)

    evals_result = {}

    model = lgb.train(params, lgtrain, 5000, 

                      valid_sets=[lgtrain, lgval], 

                      early_stopping_rounds=300, 

                      verbose_eval=100, 

                      evals_result=evals_result)

    print("asdasdasdasd")

    lgb_prediction = np.expm1(model.predict(test, num_iteration=model.best_iteration))

    return lgb_prediction, model, evals_result
lgb_pred, model, evals_result = run_lgb(x, y, x_test, y_test, test1)

print("LightGBM Training Completed...")
lgb_pred




sub_lgb = pd.DataFrame()

sub_lgb["TARGET"] = lgb_pred

sub_lgb["SK_ID_CURR"] = test1["SK_ID_CURR"]

sub_lgb.to_csv('sub_lgb_xgb_cat.csv', index=False)
import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor

def run_xgb(train_X, train_y, val_X, val_y, test_X):

    params = {'objective': 'reg:linear', 

          'eval_metric': 'rmse',

          'eta': 0.001,

          'max_depth': 10, 

          'subsample': 0.6, 

          'colsample_bytree': 0.6,

          'alpha':0.001,

          'random_state': 42, 

          'silent': True}

    

    tr_data = xgb.DMatrix(train_X, train_y)

    va_data = xgb.DMatrix(val_X, val_y)

    

    watchlist = [(tr_data, 'train'), (va_data, 'valid')]

    

    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)

    

    dtest = xgb.DMatrix(test_X)

    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))

    

    return xgb_pred_y, model_xgb
# Training XGB

pred_test_xgb, model_xgb = run_xgb(x, y, x_test, y_test, test)

print("XGB Training Completed...")
cb_model = CatBoostRegressor(iterations=500,

                             learning_rate=0.05,

                             depth=10,

                             eval_metric='RMSE',

                             random_seed = 42,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 50,

                             od_wait=20)
cb_model.fit(dev_X, dev_y,

             eval_set=(val_X, val_y),

             use_best_model=True,

             verbose=50)