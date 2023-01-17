import numpy as np 
import pandas as pd
from  datetime import datetime, timedelta
import gc
import lightgbm as lgb
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
# historical daily unit sales data per product and store
stv = pd.read_csv("../input/m5-forecasting/sales_train_validation.csv")
stv.head()
# information about the price of the products sold per store and date
sp = pd.read_csv("../input/m5-forecasting/sell_prices.csv")
sp.head()
# calendar dates info
cal = pd.read_csv("../input/m5-forecasting/calendar.csv")
cal.head()
ss = pd.read_csv("../input/m5-forecasting/sample_submission.csv")
ss.head()
stv.info()
# drop NA values from 'sales_train_validation.csv'
stv.dropna(inplace = True)
stv.shape
CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
pd.options.display.max_columns = 50
cal.tail()
h = 28 
max_lags = 57
train_last = 1913
f_day = datetime(2016,4, 25) 
f_day
def create_stv(is_train = True, nrows = None, first_day = 1500):
    sales_price = pd.read_csv("../input/m5-forecasting/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            sales_price[col] = sales_price[col].cat.codes.astype("int16")
            sales_price[col] -= sales_price[col].min()
            
    cal = pd.read_csv("../input/m5-forecasting/calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else train_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,train_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    stv = pd.read_csv("../input/m5-forecasting/sales_train_validation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            stv[col] = stv[col].cat.codes.astype("int16")
            stv[col] -= stv[col].min()
    
    if not is_train:
        for day in range(train_last+1, train_last+ 28 +1):
            stv[f"d_{day}"] = np.nan
    
    stv = pd.melt(stv,
                  id_vars = catcols,
                  value_vars = [col for col in stv.columns if col.startswith("d_")],
                  var_name = "d",
                  value_name = "sales")
    
    stv = stv.merge(cal, on= "d", copy = False)
    stv = stv.merge(sales_price, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
    return stv
def create_features(stv):
    lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        stv[lag_col] = stv[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            stv[f"rmean_{lag}_{win}"] = stv[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }
    
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in stv.columns:
            stv[date_feat_name] = stv[date_feat_name].astype("int16")
        else:
            stv[date_feat_name] = getattr(stv["date"].dt, date_feat_func).astype("int16")
FIRST_DAY = 500
%%time

stv = create_stv(is_train=True, first_day= FIRST_DAY)
stv.shape
stv.info()
stv.head()
%%time

create_features(stv)
stv.shape
stv.info()
stv.head()
stv.dropna(inplace = True)
stv.shape
cat_features = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ['event_name_1', 'event_name_2', 'event_type_1', 'event_type_2']
useless_columns = ['id', 'date', 'sales','d', 'wm_yr_wk', 'weekday']
train_columns = stv.columns[~stv.columns.isin(useless_columns)]
X_train = stv[train_columns]
y_train = stv['sales']
%%time

np.random.seed(1000)

fake_valid_inds = np.random.choice(X_train.index.values, 2_000_000, replace = False)
train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
train_data = lgb.Dataset(X_train.loc[train_inds], label = y_train.loc[train_inds], 
                         categorical_feature=cat_features, free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[train_inds], label = y_train.loc[train_inds],
                              categorical_feature=cat_features,
                 free_raw_data=False)# This is a random sample.
#train_data.savebinary('train.bin')
del stv, X_train, y_train, fake_valid_inds,train_inds ; gc.collect()
# Defining parameters for the model
params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.01,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : 100,
    'num_leaves': 100,
    "min_data_in_leaf": 100,
}
%%time
# We will use LightGBM model


m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=1)
m_lgb.save_model("model.lgb")
ss = pd.read_csv("../input/m5-forecasting/sample_submission.csv")
ss.head()
%%time

alphas = [1.028, 1.023, 1.018]
weights = [1/len(alphas)]*len(alphas)
sub = 0.

for icount, (alpha, weight) in enumerate(zip(alphas, weights)):

    te = create_stv(False)
    cols = [f"F{i}" for i in range(1,29)]

    for tdelta in range(0, 28):
        day = f_day + timedelta(days=tdelta)
        print(tdelta, day)
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
        create_features(tst)
        tst = tst.loc[tst.date == day , train_columns]
        te.loc[te.date == day, "sales"] = alpha*m_lgb.predict(tst) # magic multiplier by kyakovlev



    te_sub = te.loc[te.date >= f_day, ["id", "sales"]].copy()

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
sub.head(10)
sub.id.nunique(), sub["id"].str.contains("validation$").sum()
sub.shape