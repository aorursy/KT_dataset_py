import pandas as pd

import os

import gc

import numpy as np

import matplotlib.pyplot as plt



# ラベルエンコーダー

from sklearn import preprocessing, metrics



import numpy as np



import lightgbm as lgb



from IPython.display import display



import warnings

warnings.filterwarnings('ignore')



pd.set_option('max_columns', 500)

pd.set_option('max_rows', 500)
# ローカル用

path = os.getcwd() + "/"



# kaggle Notebook用

INPUT_DIR = '../input/m5-forecasting-accuracy'
# sales_train_validation.csv

try:

    stv = pd.read_csv(path + "sales_train_validation.csv") # ローカル用

except FileNotFoundError:

    stv = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv") # kaggle用



    

# calendar.csv

try:

    cal = pd.read_csv(path + "calendar.csv") # ローカル用

except FileNotFoundError:

    cal = pd.read_csv(f"{INPUT_DIR}/calendar.csv") # kaggle用



    

# sell_prices.csv

try:

    price = pd.read_csv(path + "sell_prices.csv") # ローカル用

except FileNotFoundError:

    price = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv") # kaggle用



    

# sample_submission.csv

try:

    ss = pd.read_csv(path + "sample_submission.csv") # ローカル用

except FileNotFoundError:

    ss = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv") # kaggle用
display(stv.head())

display(stv.tail())

display(stv.dtypes)

display(cal.head())

display(cal.tail())

display(cal.dtypes)

display(cal.max())

display(price.head())

display(price.tail())

display(price.dtypes)

display(price.max())

display(price.shape)

display(ss.head())

display(ss.tail())

display(ss.shape)
day1_1913 = [f"d_{i}" for i in range(1, 1914)]
stv_melt =  pd.melt(stv, id_vars=['id','store_id','item_id'],

           value_vars=day1_1913,

           var_name = "d", value_name = "vol")
display(stv_melt.head())

display(stv_melt.tail())

display(stv_melt.dtypes)

display(stv_melt.shape)
del day1_1913, path

gc.collect()
product = stv[["id","item_id","store_id"]]
ss_val = ss[0:30490]

ss_val.columns = ["id"] + [f"d_{d}" for d in range(1914, 1942)]



ss_eva = ss[30490:60980]

ss_eva.columns = ["id"] + [f"d_{d}" for d in range(1942, 1970)]
ss_eva['id'] = ss_eva['id'].str.replace('_evaluation','_validation')
ss_val = pd.merge(ss_val, product, how = 'left', left_on = ['id'], right_on = ['id'])

ss_eva = pd.merge(ss_eva, product, how = 'left', left_on = ['id'], right_on = ['id'])
display(ss_val.head(3))

display(ss_val.tail(3))

display(ss_val.shape)

display(ss_eva.head(3))

display(ss_eva.tail(3))

display(ss_eva.shape)
val_1914_1941 = [f"d_{i}" for i in range(1914, 1942)]

eva_1942_1969 = [f"d_{i}" for i in range(1942, 1970)]
val_melt =  pd.melt(ss_val, id_vars=['id','store_id', "item_id"],

           value_vars=val_1914_1941,

           var_name = "d", value_name = "vol")

eva_melt =  pd.melt(ss_eva, id_vars=['id','store_id', "item_id"],

           value_vars=eva_1942_1969,

           var_name = "d", value_name = "vol")
stv_melt = pd.concat([stv_melt, val_melt, eva_melt])
display(stv_melt.head(3))

display(stv_melt.tail(3))

display(stv_melt.shape)
del ss, ss_val, ss_eva, val_1914_1941, eva_1942_1969, val_melt, eva_melt, product

gc.collect()
cal = cal[["date","wm_yr_wk","d"]]
stv_melt = pd.merge(stv_melt, cal, how = 'left', left_on = ['d'], right_on = ['d'])
del cal

gc.collect()
display(stv_melt.head())

display(stv_melt.tail())

display(stv_melt.dtypes)

display(stv_melt.shape)
stv_melt = stv_melt.merge(price, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
del price

gc.collect()
display(stv_melt.head())

display(stv_melt.tail())

display(stv_melt.dtypes)

display(stv_melt.shape)
for i in [1,2,3,4]:

    stv_melt['shift%s'%i] = stv_melt["vol"].shift(i)
display(stv_melt.head(3))

display(stv_melt.tail(3))

display(stv_melt.dtypes)
stv_melt[["vol", "wm_yr_wk"]] = stv_melt[["vol", "wm_yr_wk"]].astype('int16')
stv_melt[["sell_price", "shift1"]] = stv_melt[["sell_price", "shift1"]].astype('float16')
stv_melt[["shift2", "shift3"]] = stv_melt[["shift2", "shift3"]].astype('float16')
stv_melt[["shift4"]] = stv_melt[["shift4"]].astype('float16')
# ラベルエンコーダーで store_idとitem_idを数値に変換

lbl = preprocessing.LabelEncoder()

stv_melt["store_id"] = lbl.fit_transform(stv_melt["store_id"])

stv_melt["item_id"] = lbl.fit_transform(stv_melt["item_id"])
display(stv_melt.head(3))

display(stv_melt.tail(3))

display(stv_melt.dtypes)
x_train = stv_melt[stv_melt['date'] <= '2016-03-27']

y_train = x_train['vol']

x_val   = stv_melt[(stv_melt['date'] > '2016-03-27') & (stv_melt['date'] <= '2016-04-24')]

y_val   = x_val['vol']

test    = stv_melt[(stv_melt['date'] > '2016-04-24')]
display(test.head())

display(test.tail())

display(test.dtypes)
del stv_melt

gc.collect()
features = [

    "store_id",

    "item_id",

    "wm_yr_wk",

    "sell_price",

    "shift1",

    "shift2",

    "shift3",

    "shift4"

]
params = {

    'boosting_type': 'gbdt',

    'metric': 'rmse',

    'objective': 'regression',

    'n_jobs': -1,

    'seed': 236,

    'learning_rate': 0.1,

    'bagging_fraction': 0.75,

    'bagging_freq': 10, 

    'colsample_bytree': 0.75}
train_set = lgb.Dataset(x_train[features], y_train)

val_set = lgb.Dataset(x_val[features], y_val)
# model 構築

model = lgb.train(params, train_set, num_boost_round = 100, early_stopping_rounds = 10, valid_sets = [train_set, val_set], verbose_eval = 10)
val_pred = model.predict(x_val[features])

val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))

print(f'Our val rmse score は {val_score}')
y_pred = model.predict(test[features])

test['vol'] = y_pred
predictions = test[['id', 'date', 'vol']]

predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'vol').reset_index()
display(predictions.head())

display(predictions.tail())

display(predictions.shape)
del features, params, x_train, y_train, x_val, y_val, test

gc.collect()
pre_val = predictions.iloc[:,:29]
pre_eva = pd.concat([predictions.iloc[:,0],predictions.iloc[:,29:57]], axis=1)

pre_eva['id'] = pre_eva['id'].str.replace('_validation', '_evaluation')
display(pre_val.head())

display(pre_val.tail())

display(pre_val.shape)



display(pre_eva.head())

display(pre_eva.tail())

display(pre_eva.shape)
pre_val.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

pre_eva.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]
pre_uni = pd.concat([pre_val, pre_eva], axis=0)
display(pre_uni.head())

display(pre_uni.tail())

display(pre_uni.shape)
del pre_val, pre_eva, predictions

gc.collect()
pre_uni.to_csv('submission.csv', index = False)