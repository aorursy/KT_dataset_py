# After running
! git clone --recursive https://github.com/Microsoft/LightGBM

#You can run this oneliner which will build and compile LightGBM with GPU enabled in colab:
! cd LightGBM && rm -rf build && mkdir build && cd build && cmake -DUSE_GPU=1 ../../LightGBM && make -j4 && cd ../python-package && python3 setup.py install --precompile --gpu;    
URL_calendar = "https://slavadatasets.s3.us-east-2.amazonaws.com/calendar.csv"
URL_sales_train ='https://slavadatasets.s3.us-east-2.amazonaws.com/sales_train_validation.csv'
URL_prices = 'https://slavadatasets.s3.us-east-2.amazonaws.com/sell_prices.csv' 
from  datetime import datetime, timedelta
import numpy as np, pandas as pd
import gc
import io
import dask.dataframe as dd
import lightgbm as lgb
CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
pd.options.display.max_columns = 50
def create_dt(is_train = True, nrows = None, first_day = 1200):
     
    prices = dd.read_csv(URL_prices,dtype = PRICE_DTYPES).compute()
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
        
    cal = dd.read_csv(URL_calendar,dtype = CAL_DTYPES).compute()
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
    dt = dd.read_csv(URL_sales_train, 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype).compute()
    
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
        "mday": "day",
#         "ime": "is_month_end",
#         "ims": "is_month_start",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
fday
# If you want to load all the data set it to '1' -->  Great  memory overflow  risk ! default= 350
FIRST_DAY = 1300
df = create_dt(is_train=True, first_day= FIRST_DAY)
df.shape
create_fea(df)
df.shape
df.dropna(inplace = True)
df.shape
df.info()
df.to_csv("df_F5.gzip",index=False,compression='gzip')
from google.colab import files
files.download('df_F5.gzip')
URL_df_F5_gzip = "https://slavadatasets.s3.us-east-2.amazonaws.com/df_F5.gzip"
%%time
df =dd.read_csv(URL_df_F5_gzip ,compression='gzip' ).compute()
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X = df[train_cols]
y = df["sales"]
train_cols
del df; gc.collect()
X.info()
np.random.seed(777)

fake_valid_inds = np.random.choice(X.index.values, 2_000_000, replace = False)
categorical_feature = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
       'snap_CA', 'snap_TX', 'snap_WI',  'week','quarter', 'mday']
train_inds = np.setdiff1d(X.index.values, fake_valid_inds)
X_train, y_train = X.loc[train_inds] , y.loc[train_inds]

X_valid, y_valid = X.loc[fake_valid_inds], y.loc[fake_valid_inds],
                             
# This is a random sample, we're not gonna apply any time series train-test-split tricks here!
del X, y, fake_valid_inds, train_inds ; gc.collect()
params = {
        "objective" : "poisson",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
    
    'device' : 'gpu',
    'verbosity': 1,
    #'num_iterations' : 1200,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}
model = lgb.LGBMRegressor(**params)
model
%%time

m_lgb = model.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse',
        early_stopping_rounds=5) 
model.booster_.save_model('mode.txt') 
from google.colab import files
files.download('mode.txt')
URL_F5_A_model_LGBM_00 = "https://slavadatasets.s3.us-east-2.amazonaws.com/F5_A_model_LGBM_00.txt"
#load from model:

m_lgb = lgb.Booster(model_file=URL_F5_A_model_LGBM_00)
print('Starting predicting...')
# predict
y_pred = m_lgb.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# feature importances
print('Feature importances:', list(m_lgb.feature_importances_))
# feature importances
print('Feature importances:', list(m_lgb.feature_importances_))
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
#     te_sub.loc[te.date >= fday+ timedelta(days=h), "id"] = te_sub.loc[te.date >= fday+timedelta(days=h), 
#                                                                           "id"].str.replace("validation$", "evaluation")
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
sub.head(10)
sub.id.nunique(), sub["id"].str.contains("validation$").sum()
sub.shape
#sub.to_csv("submission_LGBM_GPU.csv",index=False)
sub.to_csv("submission_LGBM_GPU.gzip",index=False,compression='gzip')
from google.colab import files
files.download('submission_LGBM_GPU.gzip')