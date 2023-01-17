from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
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
pd.options.display.max_columns = 50
data_dir = "../input/m5-forecasting-accuracy/"
calendar = pd.read_csv(data_dir+'calendar.csv')
calendar = reduce_mem_usage(calendar)
print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

sell_prices = pd.read_csv(data_dir+'sell_prices.csv')
sell_prices = reduce_mem_usage(sell_prices)
print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

sales = pd.read_csv(data_dir+'sales_train_validation.csv')
print('Sales train validation has {} rows and {} columns'.format(sales.shape[0], sales.shape[1]))
idCols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
product = sales[idCols].drop_duplicates()
import warnings
warnings.filterwarnings('ignore')

submission = pd.read_csv(data_dir+'sample_submission.csv')
validate_submission = submission[submission.id.str.endswith('validation')]
eval_submission = submission[submission.id.str.endswith('evaluation')]

# change column name
newcolumns = ["id"] + ["d_{}".format(i) for i in range(1914, 1914+28)]
validate_submission.columns = newcolumns
validate_submission = validate_submission.merge(product, how = 'left', on = 'id')

# newcolumns = ["id"] + ["d_{}".format(i) for i in range(1942, 1942+28)]
# eval_submission.columns = newcolumns
# eval_submission['id'] = eval_submission['id'].str.replace('_evaluation','_validation')
# eval_submission = eval_submission.merge(product, how = 'left', on = 'id')

idCols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

# Use only the last three years
DAYS = 365*2; LAST_DAY=1913
dayCols = ["d_{}".format(i) for i in range(LAST_DAY-DAYS+1, LAST_DAY+1)]
print(len(dayCols), dayCols[0])
sales = sales[idCols+dayCols]
print(sales.shape)
sales.head()
def melted(df, name=""):
    df = pd.melt(df, id_vars = idCols, var_name = 'day', value_name = 'demand')
    print('{}: {} rows and {} columns'.format(name, df.shape[0], df.shape[1]))
    df = reduce_mem_usage(df)
    # df.to_csv(name+".csv")
    return df

melted_sales = melted(sales)
melted_sales["part"] = "train"
melted_validate = melted(validate_submission)
melted_validate["part"] = "validate"
# melted_eval = melted(eval_submission)
# melted_eval["part"] = "evaluate"
# data = melted_sales
# data = pd.concat([melted_sales, melted_validate, melted_eval], axis = 0)
data = pd.concat([melted_sales, melted_validate], axis = 0)
data.head()
import gc
del melted_sales, melted_validate
del submission, validate_submission, eval_submission, product
del sales,
gc.collect()
# df.dropna(inplace = True)
# df.shape
# merge with calendar, sell_prices

calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
data.drop(['d', 'day'], inplace = True, axis = 1)

print('Our dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

# get the sell price data (this feature should be very important)
data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'])

print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

del calendar, sell_prices
gc.collect()
from sklearn import preprocessing
def encode_categorical(dt, cols):
    for col in cols:
        # Leave NaN as it is.
#         le = preprocessing.LabelEncoder()
#         not_null = df[col][df[col].notnull()]
#         df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)
        
#         np.save(f'label_class_{col}.npy', le.classes_)
        dt[col] = dt[col].astype("category").cat.codes.astype("int16")
        dt[col] -= dt[col].min()
    return dt


data = encode_categorical(data, ["cat_id", "dept_id", "item_id", "state_id", "store_id"]).pipe(reduce_mem_usage)
values = {'event_name_1': "normal", 'event_type_1': "normal", "event_name_2": "normal", 'event_type_2': "normal"}
data.fillna(value=values, inplace = True);

data = encode_categorical(data, ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]).pipe(reduce_mem_usage)
def datetime_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    attrs = [
        "year", 
        "quarter", 
        "month", 
        "week", 
        "day", 
        "dayofweek", 
        "weekday",
        "weekofyear",
#         "is_year_end", 
#         "is_year_start", 
#         "is_quarter_end", 
#         "is_quarter_start", 
#         "is_month_end",
#         "is_month_start",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        df[attr] = getattr(df['date'].dt, attr).astype(dtype)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    
    return df

data = datetime_features(data)
data.head()
data.sort_values(by=['id', "date"], inplace=True)

X_train = data[data["part"]=="train"]
X_val = data[data["part"]=="validate"]
X_eval = data[data["part"]=="evaluate"]

print(len(X_train), len(X_val), len(X_eval))
del data; gc.collect()
import pickle 
dbfile = open('X_train.pkl', 'wb') 
pickle.dump(X_train, dbfile)
dbfile.close() #Dont forget this 

dbfile = open('X_val.pkl', 'wb') 
pickle.dump(X_val, dbfile)
dbfile.close() #Dont forget this 
import math

def numerical_feature(df):
    for i in [7, 28]:
        df[f"shifted_t{i}"] = df[["id","demand"]].groupby('id')["demand"].shift(i)

    for win, col in [(7, "shifted_t7"), (7, "shifted_t28"), (28, "shifted_t7"), (28, "shifted_t28")]:
        df[f"rolling_mean_{col}_w{win}"] = df[["id", col]].groupby('id')[col].shift(1).rolling(win, min_periods=1).mean()
    return df


%%time

X_train = numerical_feature(X_train)
X_train.dropna(inplace = True)
gc.collect()
X_train.shape, X_train.columns
cat_feats = ['item_id', 'dept_id','store_id', 'cat_id', 'state_id'] + ["event_name_1", "event_name_2", "event_type_1", "event_type_2"]
useless_cols = ["id", "part", "date", "demand","d", "wm_yr_wk", "weekday"]
train_cols = X_train.columns[~X_train.columns.isin(useless_cols)]
train_cols 

y_train = X_train["demand"]
X_train = X_train[train_cols]

%%time
from sklearn.model_selection import train_test_split
np.random.seed(777)

X, x_test, Y, y_test = train_test_split(X_train, y_train, test_size=0.05)
del X_train, y_train; gc.collect()
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
    'num_iterations' : 3000,
    'num_leaves': 128,
    "min_data_in_leaf": 100,
}
# del X_eval, X_val


train_set = lgb.Dataset(X, Y)
test_set = lgb.Dataset(x_test, y_test)
del X, Y, x_test, y_test; gc.collect()
import pickle 
dbfile = open('train_set.pkl', 'wb') 
pickle.dump(train_set, dbfile)
dbfile.close() #Dont forget this 

dbfile = open('test_set.pkl', 'wb') 
pickle.dump(test_set, dbfile)
dbfile.close() #Dont forget this 
# from IPython.display import FileLink
# FileLink(r'train_set.pkl')
# del FileLink, dbfile, encode_categorical, math, np, pickle, preprocessing, reduce_mem_usage, train_test_split, timedelta
# gc.collect()
%%time

# model = lgb.train(params, train_set, num_boost_round = 3000, early_stopping_rounds = 500, valid_sets = [train_set, test_set], verbose_eval = 100)
model = lgb.train(params, train_set, valid_sets = [test_set], verbose_eval = 100)

print("YEAH")
model.save_model("model.lgb")
import pickle 
dbfile = open('LightGBM.pkl', 'wb') 
pickle.dump(model, dbfile)
dbfile.close() #Dont forget this 
import pickle
with open("X_train.pkl", 'rb') as fin:
    X_train = pickle.load(fin)

with open("X_val.pkl", 'rb') as fin:
    X_val = pickle.load(fin)
X_train.head()
%%time
lastdate = X_train["date"].max() - pd.DateOffset(days=86)
X_train = X_train[X_train['date'] >= lastdate]
X_train.shape, X_val.shape
import time

import datetime
import dateutil.relativedelta

def predict(model, X_train, X_test, factor=1):
    DATES = X_test["date"].unique()
    NDATE = len(DATES)
    print("NDATE", NDATE)
    
    col = ["id"] + ["F{}".format(i) for i in range(1, NDATE+1)]
    itemId = X_train["dept_id"].unique()
    print("#CHUNK", len(itemId))
    
    acc_o = []
    itemId = sorted(itemId)
    for iid in itemId:
        test = X_test[X_test["dept_id"]==iid]
        
        ids = test["id"].unique()
        oarr = np.zeros((len(ids), NDATE+1))
        o = pd.DataFrame(oarr, columns=col)
        
        o["id"] = test[test["date"]==DATES[0]]["id"].values
        
        train = X_train[X_train["dept_id"]==iid]
        
        ## XX=test, X=train
        lastmonth = pd.to_datetime(train.head(1)["date"])
        for idx, date in enumerate(DATES):
            
            newrow = test[test["date"]==date]
            train = train.append(newrow)
            
            train.sort_values(by=['id', "date"], inplace=True)
# #             print("num feats START")
            feat = numerical_feature(train)
            
# #             print("num feats DONE")

#             print(f"============== {idx} ==========")
#             p = feat[feat["id"]=="FOODS_1_001_CA_1_validation"]
# #             print(p)
#             print(p.tail(15)[["date", "demand", "shifted_t7"]])
#             if idx==10:
#                 return None
            
            x = feat.loc[feat["date"] == date , train_cols]
            val_pred = model.predict(x, num_iteration=model.best_iteration)

            
            o[f"F{idx+1}"] = val_pred*factor
            
            
            train.loc[train["date"]==date, "demand"] = val_pred*factor
            
            
            lastmonth = lastmonth + pd.DateOffset(days=1)
            train = train[train['date'] >= str(lastmonth.values[0])]
            
        acc_o.append(o)
        acc_o = [pd.concat(acc_o)]
        print(iid)
#         break
    
    acc_o = pd.concat(acc_o)
    return acc_o
pp = []
weights = [1, 1.028, 1.023, 1.018]
for w in weights:
    print("======== w",w,"==========")
    pred = predict(model, X_train, X_val, factor=w)
    pp.append(pred)
avgpred = pd.DataFrame([])
avgpred["id"] = pp[0]["id"]
for i in range(1, 29):
    avgpred[f"F{i}"] = (pp[1][f"F{i}"]+pp[2][f"F{i}"]+pp[3][f"F{i}"])/3
    
#     print(sum(avgpred[f"F{i}"]), sum(pp[0][f"F{i}"]))
    

submission = pd.read_csv(data_dir+'sample_submission.csv')
dfeval = submission[submission.id.str.endswith('evaluation')]
assert len(dfeval)==len(avgpred)
def save_csv(i, d):
    df = pd.concat([d, dfeval]) 

    df.sort_values("id", inplace = True)
    df.reset_index(drop=True, inplace = True)
    if i=="":
        df.to_csv(f"submission.csv")
    else:
        df.to_csv(f"submission.v9.{i}.csv")
    
for i in range(len(pp)):
    save_csv(i, pp[i])

save_csv("", avgpred)
