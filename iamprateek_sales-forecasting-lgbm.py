# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 

         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",

        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)

calendar_df = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)

sales_train_validation = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

subm = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
sell_prices.shape
calendar_df.shape
sales_train_validation.shape
sell_prices.head()
calendar_df.head()
sales_train_validation.head()
subm.head()
sales_train_validation.columns
sales_train_validation.info()
from  datetime import datetime, timedelta

import gc
h = 28 

max_lags = 57

tr_last = 1913

fday = datetime(2016,4, 25)
def create_dt(is_train = True, nrows = None, first_day = 1200):

    prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)

    for col, col_dtype in PRICE_DTYPES.items():

        if col_dtype == "category":

            prices[col] = prices[col].cat.codes.astype("int16")

            prices[col] -= prices[col].min()

            

    cal = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)

    cal["date"] = pd.to_datetime(cal["date"])

    for col, col_dtype in CAL_DTYPES.items():

        if col_dtype == "category":

            cal[col] = cal[col].cat.codes.astype("int16")

            cal[col] -= cal[col].min()

    

    start_day = max(1 if is_train  else tr_last-max_lags, first_day)

    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]

    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    dtype = {numcol:"float32" for numcol in numcols} 

    dtype.update({col: "category" for col in catcols if col != "id"})

    dt = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv", 

                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)

    

    for col in catcols:

        if col != "id":

            dt[col] = dt[col].cat.codes.astype("int16")

            dt[col] -= dt[col].min()

    

    if not is_train:

        for day in range(tr_last+1, tr_last+ 28 +1):

            dt[f"d_{day}"] = np.nan

    

    dt = pd.melt(dt,

                  id_vars = catcols,

                  value_vars = [col for col in dt.columns if col.startswith("d_")],

                  var_name = "d",

                  value_name = "sales")

    

    dt = dt.merge(cal, on= "d", copy = False)

    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)

    

    return dt
def create_fea(dt):

    lags = [7, 28]

    lag_cols = [f"lag_{lag}" for lag in lags ]

    for lag, lag_col in zip(lags, lag_cols):

        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)



    wins = [7, 28]

    for win in wins :

        for lag,lag_col in zip(lags, lag_cols):

            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())



    

    

    date_features = {

        

        "wday": "weekday",

        "week": "weekofyear",

        "month": "month",

        "quarter": "quarter",

        "year": "year",

        "mday": "day"

    }

    

    for date_feat_name, date_feat_func in date_features.items():

        if date_feat_name in dt.columns:

            dt[date_feat_name] = dt[date_feat_name].astype("int16")

        else:

            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
FIRST_DAY = 350 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !
%%time



df = create_dt(is_train=True, first_day= FIRST_DAY)

df.shape
df.head()
df.info()
%%time



create_fea(df)

df.shape
df.dropna(inplace = True)

df.shape
cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]

train_cols = df.columns[~df.columns.isin(useless_cols)]

X_train = df[train_cols]

y_train = df["sales"]
import lightgbm as lgb
%%time



np.random.seed(777)



fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)

train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)

train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], 

                         categorical_feature=cat_feats, free_raw_data=False)

fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],

                              categorical_feature=cat_feats,

                 free_raw_data=False)# This is a random sample, we're not gonna apply any time series train-test-split tricks here!
# release some memory

del df, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()
params = {

        "objective" : "poisson",

        "metric" :"rmse",

        "force_row_wise" : True,

        "learning_rate" : 0.075,

        "sub_row" : 0.75,

        "bagging_freq" : 1,

        "lambda_l2" : 0.1,

        "metric": ["rmse"],

        'verbosity': 1,

        'num_iterations' : 1200,

        'num_leaves': 128,

        "min_data_in_leaf": 100,

}
%%time



m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=20) 
# save the model

m_lgb.save_model("save_model.lgb")
%%time



alphas = [1.028, 1.023, 1.018]

weights = [1/len(alphas)]*len(alphas)

sub = 0.



for icount, (alpha, weight) in enumerate(zip(alphas, weights)):



    te = create_dt(False)

    cols = [f"F{i}" for i in range(1,29)]



    for tdelta in range(0, 28):

        day = fday + timedelta(days=tdelta)

        print(tdelta, day)

        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()

        create_fea(tst)

        tst = tst.loc[tst.date == day , train_cols]

        te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev







    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()

    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")["id"].cumcount()+1]

    te_sub = te_sub.set_index(["id", "F" ]).unstack()["sales"][cols].reset_index()

    te_sub.fillna(0., inplace = True)

    te_sub.sort_values("id", inplace = True)

    te_sub.reset_index(drop=True, inplace = True)

    te_sub.to_csv(f"submission_{icount}.csv",index=False)

    if icount == 0 :

        sub = te_sub

        sub[cols] *= weight

    else:

        sub[cols] += te_sub[cols]*weight

    print(icount, alpha, weight)





sub2 = sub.copy()

sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")

sub = pd.concat([sub, sub2], axis=0, sort=False)

sub.to_csv("submission.csv",index=False)
sub.head()