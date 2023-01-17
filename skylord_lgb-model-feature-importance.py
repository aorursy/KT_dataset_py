import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import lightgbm as lgb
# Load the models that were trained 



model_0x48874 = lgb.Booster(model_file='/kaggle/input/m5-forecasting-models/model_0x48874_.lgb')

model_0x48743 = lgb.Booster(model_file='/kaggle/input/m5-forecasting-models/model_0x48743_.lgb')

model = lgb.Booster(model_file='/kaggle/input/m5-forecasting-models/model.lgb')
from  datetime import datetime, timedelta

import gc
# Load the data 



CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 

         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",

        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }

PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }



h = 28 

max_lags = 70

tr_last = 1913

fday = datetime(2016,4, 25) 



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

            

FIRST_DAY = 750 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !



df = create_dt(is_train=True, first_day= FIRST_DAY)

print(df.shape)



create_fea(df)

print(df.shape)



df.dropna(inplace = True)



cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]

useless_cols = ["id", "date", "sales","d", "wm_yr_wk", "weekday"]

train_cols = df.columns[~df.columns.isin(useless_cols)]

X_train = df[train_cols]

y_train = df["sales"]
# import graph objects as "go" and import tools

import plotly.graph_objs as go

from plotly import tools



import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

init_notebook_mode(connected=True)
model.importance()




# sorted(zip(clf.feature_importances_, X.columns), reverse=True)

feature_imp = pd.DataFrame(zip(model.feature_importance(importance_type='gain', iteration=-1), model.feature_name()),  columns=['Value_Gain','Feature'])

feature_imp['Value_Split'] = model.feature_importance(importance_type='split', iteration=-1)

# plt.figure(figsize=(20, 10))

# sns.barplot(x="Value_Gain", y="Feature", data=feature_imp.sort_values(by="Value_Gain", ascending=False))

# plt.title('LightGBM Features (Gain)')

# plt.tight_layout()

# plt.show()

# plt.savefig('lgbm_importances-01-gain.png')



# plt.figure(figsize=(20, 10))

# sns.barplot(x="Value_Split", y="Feature", data=feature_imp.sort_values(by="Value_Split", ascending=False))

# plt.title('LightGBM Features (Split)')

# plt.tight_layout()

# plt.show()

# plt.savefig('lgbm_importances-01-split.png')

feature_imp.sort_values(by = ['Value_Gain'], ascending = True, inplace = True)



# create trace1 

trace1 = go.Bar(

                y=feature_imp['Feature'],

                x=feature_imp['Value_Gain'],

                name = "feature_importance_gain",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                orientation='h',

                #xaxis = 'x1',

                #yaxis = 'y1',

                

                text = feature_imp['Feature'])



data = [trace1]

layout = go.Layout(

    barmode = "group", title="Feature Importance by Gain" )



fig = go.Figure(data = data, layout = layout)

iplot(fig)
feature_imp.sort_values(by = ['Value_Split'], ascending = True, inplace = True)



trace2 = go.Bar(

                y=feature_imp['Feature'],

                x=feature_imp['Value_Split'],

                name = "feature_importance_split",

                marker = dict(color = 'rgba(174, 255, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                orientation='h',

                #xaxis = 'x2',

                #yaxis = 'y2',

                text = feature_imp['Feature'])



data = [trace2]

layout = go.Layout(

    barmode = "group", title="Feature Importance by Split" )

fig = go.Figure(data = data, layout = layout)

iplot(fig)