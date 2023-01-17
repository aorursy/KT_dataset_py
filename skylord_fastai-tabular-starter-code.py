from  datetime import datetime, timedelta

import time

import gc

import numpy as np, pandas as pd

import gc
from fastai import *      # import * is considered as a bad coding practice! will have to change this!

from fastai.tabular import *
start_nb = time.time()
CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 

         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",

        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }



pd.options.display.max_columns = 50

h = 28 

max_lags = 70

tr_last = 1913

fday = datetime(2016,4, 25) 
def create_dt(is_train = True, nrows = None, first_day = 1200, store_id = None):

    

    start = time.time()

    if store_id == None and is_train:

        print("ERROR: No store_id provided.Please provide an id [0-9]")

        return None

        

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

    

    # Filter out the values for store_id

    if is_train:

        prices = prices[prices['store_id'] == store_id]

        

    print(f"Shape of Store - {store_id} dataframe : ", prices.shape)

    

    

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

    end = time.time()

    

    print("Processing time: ", (end-start))

    return dt
def create_fea(dt):

    start = time.time()

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

    

    # Drop NA values

    dt.dropna(inplace = True)     

    

    # Sort the dataframe on 'saledate' so we can easily create a validation set that data is in the 'future' of what's in the training set

    dt = dt.sort_values(by='date', ascending=False)

    dt = dt.reset_index(drop=True)

    end = time.time()

    print("Processing time: ", (end- start))

    

    return dt
import shutil



def checkDiskSpace():

    total, used, free = shutil.disk_usage("/")

    

    free = (free // (2**30))

    

    if free < 10:

        return -1

    else:

        return 0 

         



# print("Total: %d GiB" % (total // (2**30)))

# print("Used: %d GiB" % (used // (2**30)))

# print("Free: %d GiB" % (free // (2**30)))
cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]

#train_cols = df.columns[~df.columns.isin(useless_cols)]
FIRST_DAY = 1 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !
idx = 1

train_df = create_dt(is_train=True, first_day= FIRST_DAY, store_id = idx)

train_df = create_fea(train_df)

print(train_df.shape)

print(train_df['date'].min(), train_df['date'].max())
# Sort by date (used for train/validation splits)

train_df.sort_values(by='date', inplace=True)



# convert sales value to log scale

train_df['sales'] = np.log(train_df['sales'] + 1)  # Taking logarithm values for sales



# Calculate where we should cut the validation set. We pick the most recent 'n' records in training set 

# where n is the number of entries in test set. 



cut = train_df['date'][(train_df['date'] == train_df['date'][62500])].index.max()

print(cut)

valid_idx = range(cut)

# Define categorical, continous & dependent variables



cat_vars = ['wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'event_name_1',

           'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX',

           'snap_WI', 'week', 'quarter', 'mday']



cont_vars = ['lag_7', 'lag_28', 'rmean_7_7', 'rmean_28_7',

           'rmean_7_28', 'rmean_28_28', 'sell_price']



dep_var = 'sales'

# We want to limit the price range for our prediction to be within the history sale price range, so we need to calculate the y_range

# Note that we multiplied the maximum of 'SalePrice' by 1.2 so when we apply sigmoid, the upper limit will also be covered. 

max_y = np.max(train_df['sales'])*1.2

y_range = torch.tensor([0, max_y], device=defaults.device)

print(y_range)


# Defining pre-processing we want for our fast.ai DataBunch

procs=[FillMissing, Categorify, Normalize]



# Use fast.ai datablock api to put our training data into the DataBunch, getting ready for training

data = (TabularList.from_df(train_df, cat_names=cat_vars, cont_names=cont_vars, procs=procs)

                       .split_by_idx(valid_idx)

                       .label_from_df(dep_var)

                       .databunch())





# Create our tabular learner. The dense layer is 1000 and 500 two layer NN. We used dropout, hai 

learn = tabular_learner(data, layers=[512,256, 128], ps=[0.05,0.01, 0.5], emb_drop=0.04, 

                            y_range=y_range, metrics=rmse)

learn.lr_find()
# Plot the learning rates

learn.recorder.plot()
# learn.fit_one_cycle(3, 1e-2, wd=0.2)

    

# print(f"Saving model...export_{idx}")

# learn.export(file = Path(f"/kaggle/working/export_{idx}.pkl")) # Save the model

learn = load_learner('/kaggle/input/m5-forecasting-models/', file=f'export_{idx}.pkl')
%%time

te = create_dt(False)

te = create_fea(te)
print(te.shape)

te.head()
learn.predict(te.loc[1])[1]
print("Raw prediction: ",learn.predict(te.loc[1]))

print("Taking exponentials: ",(np.exp(learn.predict(te.loc[1])[1]) -1) )

end_nb = time.time()



print("Notebook processing time: ", (end_nb- start_nb))