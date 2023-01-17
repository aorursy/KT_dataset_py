from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
from catboost import Pool, CatBoostRegressor
#import lightgbm as lgb
import pandas as pd
import shap
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
pd.options.display.max_columns = 50
CAL_DTYPES={"weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16"}
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
fday
def create_dt(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype = CAL_DTYPES)
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
## modified from the original
def create_fea(dt, lags, windows):
    print("creating window: ", windows, " lag: ", lags)
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    for win in windows :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())


    ## drop lag col
    dt.drop(lag_cols, axis=1, inplace=True)
    
    
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
#FIRST_DAY = 350 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !
# I used 350 as the first day 
# To avoid maxing out the memory
FIRST_DAY = 1000
%%time

df = create_dt(is_train=True, first_day= FIRST_DAY)
df.columns
df.info()
df.drop(['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI'], axis=1,\
       inplace=True)
gc.collect()
%%time
create_fea(df, lags=[7,14, 28, 30, 50, 60], windows=[7, 28])
df.shape
df.info()
# X_train.corr()
df.dropna(inplace = True)
df.shape
cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id']
useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]
train_cols = df.columns[~df.columns.isin(useless_cols)]
X_train = df[train_cols]
y_train = df["sales"]
X_train.shape
%%time

np.random.seed(777)

fake_valid_inds = np.random.choice(X_train.index.values, int(X_train.shape[0]*.2), replace = False)
train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
fake_valid_inds_2 = np.random.choice(X_train.index.values, int(X_train.shape[0]*.002), replace = False)

import seaborn as sns
plt.figure(figsize=(10, 10))
sns.heatmap(X_train.loc[fake_valid_inds_2].corr())
gc.collect()
## load model
from catboost import Pool, CatBoostRegressor

from_file = CatBoostRegressor()
model = from_file.load_model('../input/catboost-baseline/catboost_model_0430')

%%time

####### I uploaded the model to avoid memory error
cat_feats_ind = [i for (i, j) in enumerate(X_train.columns) if j in cat_feats]

# # put in pool
train_pool = Pool(X_train.loc[train_inds], y_train.loc[train_inds], cat_features=cat_feats_ind)
test_pool = Pool(X_train.loc[fake_valid_inds], y_train.loc[fake_valid_inds], cat_features=cat_feats_ind)

# # train
# # I used iteration = 80 and depths = 3 at first
# model =CatBoostRegressor(iterations=80, \
#                       depth=3, \
#                       learning_rate=1, \
#                       loss_function='RMSE',l2_leaf_reg=1)


###############################################################


#model.fit(train_pool, eval_set=test_pool, verbose=False)

# plot
plt.style.use('fivethirtyeight')
plt.figure(figsize=(5,10))
plt.barh(train_pool.get_feature_names(), model.get_feature_importance())
plt.xticks(rotation=90)
plt.show()

gc.collect()
feature_df = pd.DataFrame({'Feature': X_train.columns, "Imp": model.get_feature_importance()})
feature_df.sort_values('Imp', ascending=False)
cat_feats_ind = [i for (i, j) in enumerate(X_train.columns) if j in cat_feats]
cat_feats_ind
gc.collect()
def create_lag_features_for_test(dt, day, lags, windows):
    # modified the original 
    # create lag feaures just for single day (faster)
    #lags = [7, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        dt.loc[dt.date == day, lag_col] = \
            dt.loc[dt.date ==day-timedelta(days=lag), 'sales'].values  # !!! main

    #windows = [7, 28]
    for window in windows:
        for lag in lags:
            df_window = dt[(dt.date <= day-timedelta(days=lag)) & (dt.date > day-timedelta(days=lag+window))]
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(dt.loc[dt.date==day,'id'])
            dt.loc[dt.date == day,f"rmean_{lag}_{window}"] = \
                df_window_grouped.sales.values     
def create_date_features_for_test(dt):
    # copy of the code from `create_dt()` above
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(
                dt["date"].dt, date_feat_func).astype("int16")
%%time

alphas = [1.028, 1.023, 1.018]
weights = [1/len(alphas)]*len(alphas)  # equal weights

te0 = create_dt(False)  # create master copy of `te`
create_date_features_for_test (te0)
model = model
for icount, (alpha, weight) in enumerate(zip(alphas, weights)):
    te = te0.copy()  # just copy
    cols = [f"F{i}" for i in range(1, 29)]

    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day.date())
        tst = te[(te.date >= day - timedelta(days=max_lags))
                 & (te.date <= day)].copy()
#         create_fea(tst)  # correct, but takes much time
        create_lag_features_for_test(tst, day, lags=[7], windows=[7])  # faster  
        tst = tst.loc[tst.date == day, train_cols]
        
        ## put tst in pool
        tst_pool = Pool(data=tst, cat_features=cat_feats_ind)
        te.loc[te.date == day, "sales"] = \
            alpha * model.predict(tst_pool)  # magic multiplier by kyakovlev

    te_sub = te.loc[te.date >= fday, ["id", "sales"]].copy()

    te_sub["F"] = [f"F{rank}" for rank in te_sub.groupby("id")[
        "id"].cumcount()+1]
    te_sub = te_sub.set_index(["id", "F"]).unstack()[
        "sales"][cols].reset_index()
    te_sub.fillna(0., inplace=True)
    te_sub.sort_values("id", inplace=True)
    te_sub.reset_index(drop=True, inplace=True)
    te_sub.to_csv(f"submission_{icount}.csv", index=False)
    if icount == 0:
        sub = te_sub
        sub[cols] *= weight
    else:
        sub[cols] += te_sub[cols]*weight
    
sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation")
sub = pd.concat([sub, sub2], axis=0, sort=False)
sub.to_csv("submission_catboost_add_iteration_04_27.csv",index=False)
#print(icount, alpha, weight)
del sub,sub2

gc.collect()
shap_values = model.get_feature_importance(tst_pool,type="ShapValues")
expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]

# we used one of the test data for shap value calculation
shap.initjs()
shap.force_plot(expected_value, shap_values[3,:], tst.iloc[3,:])
%%time
shap.summary_plot(shap_values, tst, plot_type='bar')

feature = ['rmean_7_7', 'rmean_7_28', 'rmean_60_7', 'rmean_50_7', 'rmean_50_28',\
           'rmean_14_7','rmean_14_28', 'rmean_28_7']
X_train.shape
model.calc_feature_statistics(X_train[23940000:], y_train[23940000:],feature=feature, plot=True)


