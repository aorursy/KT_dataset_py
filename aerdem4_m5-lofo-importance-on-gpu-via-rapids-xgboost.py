import sys

!cp ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
!git clone https://github.com/aerdem4/rapids-kaggle-utils.git

%cd rapids-kaggle-utils/
!pip install -U xgboost
from  datetime import datetime, timedelta

import gc

import numpy as np, pandas as pd

import lightgbm as lgb



import cudf

import cu_utils.transform as cutran
h = 28 

max_lags = 57

tr_last = 1913

fday = datetime(2016,4, 25) 

FIRST_DAY = 1000

fday
%%time



def create_df(start_day):

    prices = cudf.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

            

    cal = cudf.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

    cal["date"] = cal["date"].astype("datetime64[ms]")

    

    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]

    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    dt = cudf.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv", usecols = catcols + numcols)

    

    dt = cudf.melt(dt,

                  id_vars = catcols,

                  value_vars = [col for col in dt.columns if col.startswith("d_")],

                  var_name = "d",

                  value_name = "sales")

    

    dt = dt.merge(cal, on= "d")

    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"])

    

    return dt





df = create_df(FIRST_DAY)
%%time



def transform(data):

    

    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    for feature in nan_features:

        data[feature].fillna('unknown', inplace = True)

    

    data['id_encode'], _ = data["id"].factorize()

    

    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']

    for feature in cat:

        data[feature], _ = data[feature].factorize()

    

    return data

        

        

df = transform(df)
%%time



def create_fea(data):



    for lag in [7, 28]:

        out_col = "lag_{}".format(str(lag))

        data[out_col] = data[["id", "sales"]].groupby("id", method='cudf').apply_grouped(cutran.get_cu_shift_transform(shift_by=lag),

                                                                      incols={"sales": 'x'},

                                                                      outcols=dict(y_out=np.float32),

                                                                      tpb=32)["y_out"]

    

        for window in [7, 28]:

            out_col = "rmean_{lag}_{window}".format(lag=lag, window=window)

            data[out_col] = data[["id", "lag_{}".format(lag)]].groupby("id", method='cudf').apply_grouped(cutran.get_cu_rolling_mean_transform(window),

                                                                          incols={"lag_{}".format(lag): 'x'},

                                                                          outcols=dict(y_out=np.float32),

                                                                          tpb=32)["y_out"]



    # time features

    data['date'] = data['date'].astype("datetime64[ms]")

    data['year'] = data['date'].dt.year

    data['month'] = data['date'].dt.month

    data['day'] = data['date'].dt.day

    data['dayofweek'] = data['date'].dt.weekday

    

    

    return data





    



# define list of features

features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',

            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 

            'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 

            'year', 'month', 'day', 'dayofweek',

            'lag_7', 'lag_28', 'rmean_7_7', 'rmean_7_28', 'rmean_28_7', 'rmean_28_28'

           ]





df = create_fea(df)

df.tail()
!pip install lofo-importance
from lofo import LOFOImportance, Dataset, plot_importance

from sklearn.model_selection import KFold

import xgboost



sample_df = df.to_pandas().sample(frac=0.2, random_state=0)

sample_df.sort_values("date", inplace=True)



cv = KFold(n_splits=7, shuffle=False, random_state=0)



dataset = Dataset(df=sample_df, target="sales", features=features)



# define the validation scheme and scorer

params = {"objective": "count:poisson",

          "learning_rate" : 0.075,

          "max_depth": 8,

          'n_estimators': 200,

          'min_child_weight': 50,

          "tree_method": 'gpu_hist', "gpu_id": 0}

xgb_reg = xgboost.XGBRegressor(**params)

lofo_imp = LOFOImportance(dataset, cv=cv, scoring="neg_mean_squared_error", model=xgb_reg)



# get the mean and standard deviation of the importances in pandas format

importance_df = lofo_imp.get_importance()
plot_importance(importance_df, figsize=(12, 12))