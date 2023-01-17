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
import numpy as np

import pandas as pd

import os

# custom imports

from multiprocessing import Pool        # Multiprocess Runs

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import gc

import statsmodels.api as sm

from pylab import rcParams

rcParams['figure.figsize'] = 18,8
# Memory reduction helper function:

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns: #columns

        col_type = df[col].dtypes

        if col_type in numerics: #numerics

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
calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", sep= ",") 

calendar = reduce_mem_usage(calendar)
calendar.shape
calendar.head(3), calendar.tail(3)
calendar['date'] = pd.to_datetime(calendar['date'], format='%Y-%m-%d')

calendar['year'] = calendar['date'].dt.year

calendar['month'] = calendar['date'].dt.month

calendar['week'] = calendar['date'].dt.week

calendar['day'] = calendar['date'].dt.day
calendar['event_type_2'][calendar['event_type_2'].isnull()] = 0

calendar['event_name_2'][calendar['event_name_2'].isnull()] = 0

calendar['event_type_1'][calendar['event_type_1'].isnull()] = 0

calendar['event_name_1'][calendar['event_name_1'].isnull()] = 0
calendar['num_events'] = pd.Series()

calendar['num_events'][(calendar['event_type_2'] != 0)] = 2

calendar['num_events'][(calendar['event_type_2'] == 0) & (calendar['event_type_1'] != 0)] = 1
calendar['num_events'][(calendar['num_events'].isnull())] = 0
calendar['is_weekend'] = pd.Series()

calendar['is_weekend'][(calendar['weekday'] == 'Saturday') | (calendar['weekday'] == 'Sunday')] = 1

calendar['is_weekend'][(calendar['weekday'] != 'Saturday') & (calendar['weekday'] != 'Sunday')] = 0
calendar.drop(['event_type_2','event_name_2','event_name_1','event_type_1','weekday'],axis=1, inplace = True)
calendar.columns
calendar.head(5)
calendar.dtypes
sprice = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", sep = ',')

sprice = reduce_mem_usage(sprice)
sprice.shape, sprice.head(5)
for col in ['store_id','item_id']:

    sprice[col] = sprice[col].astype('category')
stv = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

stv = reduce_mem_usage(stv)



stv.shape
stv.head(5)
stv.drop(['id','dept_id','cat_id','state_id'],axis = 1, inplace = True)
stv.columns
# Define all categorical columns

catCols = ['item_id', 'store_id']

stv = pd.melt(stv,

             id_vars = catCols,

             value_vars = [col for col in stv.columns if col.startswith("d_")],

             var_name = "d",

             value_name = "Demand")
stv = reduce_mem_usage(stv)
merge1 = stv.merge(calendar, on = "d", copy = False)
merge1 = reduce_mem_usage(merge1)
ds = merge1.merge(sprice, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
ds.head()
ds.drop(['d','wm_yr_wk'],axis = 1, inplace = True)
test = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
test.columns = ['id','d_1914','d_1915','d_1916','d_1917','d_1918','d_1919','d_1920',

               'd_1921','d_1922','d_1923','d_1924','d_1925','d_1926','d_1927','d_1928','d_1929','d_1930',

               'd_1931','d_1932','d_1933','d_1934','d_1935','d_1936','d_1937','d_1938','d_1939','d_1940','d_1941']
catCols = ['id']

test = pd.melt(test,

             id_vars = catCols,

             value_vars = [col for col in test.columns if col.startswith("d_")],

             var_name = "d",

             value_name = "Demand")
test_c = test.merge(calendar, on = "d", copy = False)
test_c['item_id'] = test_c['id'].str.slice(0,-16)

test_c['store_id'] = test_c['id'].str.slice(-15,-11)
test_f = test_c.merge(sprice, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
test_p = test_f.drop(['id','d','Demand','wm_yr_wk'],axis = 1)
del stv, merge1

gc.collect()
ds.isnull().sum()
test_f.dtypes
ds.shape, test_p.shape
ds['store_id'].value_counts()
train = ds[ds['date'] < '2016-01-01']

val = ds[ds['date'] >= '2016-01-01']
train = train.set_index('date')

val = val.set_index('date')

test_p = test_p.set_index('date')
cols = ['item_id','wday','month','year','snap_CA','snap_TX','snap_WI','week','day','num_events','is_weekend']



for col in cols:

    train[col] = train[col].astype('category')

    val[col] = val[col].astype('category')

    test_p[col] = test_p[col].astype('category')
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
train['Pred_Demand'] = pd.Series()

val['Pred_Demand'] = pd.Series()
storeid = ['CA_1','CA_2','CA_3','CA_4','WI_1','WI_2','WI_3','TX_1','TX_2','TX_3']



for i in storeid:

    train_i = train[train['store_id'] == i]

    val_i = val[val['store_id'] == i]

    test_i = test_p[test_p['store_id'] == i]

    

    x_train = train_i.copy().drop(['Demand','store_id','Pred_Demand'],axis = 1)

    y_train = train_i['Demand']

    

    x_val = val_i.copy().drop(['Demand','store_id','Pred_Demand'], axis = 1)

    y_val = val_i['Demand']

    

    test = test_i.drop(['store_id'],axis = 1)

    

    print(i)

    print(x_train.shape,x_val.shape,test.shape)

    

    x_train['item_id'] = x_train['item_id'].cat.codes

    x_val['item_id'] = x_val['item_id'].cat.codes

    test['item_id'] = test['item_id'].cat.codes

    

    num_col = ['sell_price']

    scale = StandardScaler()

    scale.fit(x_train[num_col])

    x_train[num_col] = scale.transform(x_train[num_col])

    x_val[num_col] = scale.transform(x_val[num_col])

    test[num_col] = scale.transform(test[num_col])

    

    regressor = DecisionTreeRegressor(max_depth=15,max_features=10,min_samples_split=2,min_samples_leaf=1)

    

    regressor.fit(x_train,y_train)

    

    train_pred = regressor.predict(x_train)

    val_pred = regressor.predict(x_val)

    test_pred = regressor.predict(test)

    

    dt_mse_train = mean_squared_error(train_pred, y_train)

    dt_rmse_train = np.sqrt(dt_mse_train)

    print("DT Regression MSE on train: %.4f" %dt_mse_train)

    print('DT Regression RMSE on train: %.4f' % dt_rmse_train)

    dt_mse_val = mean_squared_error(val_pred, y_val)

    dt_rmse_val = np.sqrt(dt_mse_val)

    print("DT Regression MSE on val: %.4f" %dt_mse_val)

    print('DT Regression RMSE on val: %.4f' % dt_rmse_val)

    

    test_f['Demand'][test_f['store_id'] == i] = test_pred

    train['Pred_Demand'][train['store_id'] == i] = train_pred

    val['Pred_Demand'][val['store_id'] == i] = val_pred
dt_mse_train = mean_squared_error(train['Pred_Demand'], train['Demand'])

dt_rmse_train = np.sqrt(dt_mse_train)

print("DT Regression MSE on train: %.4f" %dt_mse_train)

print('DT Regression RMSE on train: %.4f' % dt_rmse_train)

dt_mse_val = mean_squared_error(val['Pred_Demand'], val['Demand'])

dt_rmse_val = np.sqrt(dt_mse_val)

print("DT Regression MSE on val: %.4f" %dt_mse_val)

print('DT Regression RMSE on val: %.4f' % dt_rmse_val)
selected_columns = test_f[['id','d','Demand']]

submission = selected_columns.copy()
submission = submission.pivot(index='id', columns='d')
submission.columns = [''] * len(submission.columns)
submission.shape, test.shape
submission.reset_index(level=0, inplace=True)
test = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')

submission.columns = test.columns
submission.head()
submission.to_csv('submission.csv')