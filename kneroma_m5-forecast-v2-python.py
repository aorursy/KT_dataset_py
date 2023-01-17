from  datetime import datetime, timedelta

import gc

import numpy as np, pandas as pd

import lightgbm as lgb
CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 

         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",

        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
pd.options.display.max_columns = 50
h = 28 

max_lags = 70

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
FIRST_DAY = 800 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !

# FIRST_DAY = 1700
%%time



df = create_dt(is_train=True, first_day= FIRST_DAY)

df.shape
df.head()
df.info()
%%time



create_fea(df)

df.shape
df.info()
df.head()
df.dropna(inplace = True)

df.shape
cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]

train_cols = df.columns[~df.columns.isin(useless_cols)]

X_train = df[train_cols]

y_train = df["sales"]
train_data = lgb.Dataset(X_train, label = y_train, categorical_feature=cat_feats, free_raw_data=False)

fake_valid_inds = np.random.choice(len(X_train), 1000000)

fake_valid_data = lgb.Dataset(X_train.iloc[fake_valid_inds], label = y_train.iloc[fake_valid_inds],categorical_feature=cat_feats,

                             free_raw_data=False)   # This is just a subsample of the training set, not a real validation set !
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

        "metric": ["rmse"],

    'verbosity': 1,

#     'num_iterations' : 200,

    'num_iterations' : 2500,

}
%%time



m_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=100) 
m_lgb.save_model("model.lgb")
%%time



alphas = [1.035, 1.03, 1.025, 1.02]

weights = [1/len(alphas)]*len(alphas)

sub = 0.



for icount, (alpha, weight) in enumerate(zip(alphas, weights)):



    te = create_dt(False)

    cols = [f"F{i}" for i in range(1,29)]



    for tdelta in range(0, 28):

#     for tdelta in range(0, 2):

        day = fday + timedelta(days=tdelta)

        print(icount, day)

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

#     te_sub.to_csv("submission.csv",index=False)

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