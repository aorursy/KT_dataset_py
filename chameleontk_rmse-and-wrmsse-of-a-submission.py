import numpy as np
import pandas as pd
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns毎に処理
        col_type = df[col].dtypes
        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更
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

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
calendar = reduce_mem_usage(calendar)
print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sell_prices = reduce_mem_usage(sell_prices)
print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))

sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
print('Sales train validation has {} rows and {} columns'.format(sales.shape[0], sales.shape[1]))
sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
sample_submission = sample_submission[sample_submission.id.str.endswith('validation')]

NUM_ITEMS = sales.shape[0]    # 30490
DAYS_PRED = sample_submission.shape[1] - 1    # 28

# To make it simpler, I will run only the last 10 days
DAYS_PRED = 10


def transform(df):
    newdf = df.melt(id_vars=["id"], var_name="d", value_name="sale")
    newdf.sort_values(by=['id', "d"], inplace=True)
    newdf.reset_index(inplace=True)
    return newdf

from sklearn.metrics import mean_squared_error

def rmse(df, gt):
    df = transform(df)
    gt = transform(gt)
    return mean_squared_error(df["sale"], gt["sale"])


dayCols = ["d_{}".format(i) for i in range(1904, 1904+DAYS_PRED)]
gt = sales[["id"]+dayCols]

dayCols = ["F{}".format(i) for i in range(1, DAYS_PRED+1)]
df = sample_submission[["id"]+dayCols]
# RSME score
rmse(df, gt)
import gc
from scipy.sparse import csr_matrix
# Get list of all products
idcols = ["id", "item_id", "state_id", "store_id", "cat_id", "dept_id"]
product = sales[idcols]

# create weight matrix
pd.get_dummies(product.state_id, drop_first=False)
weight_mat = np.c_[
   np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
   pd.get_dummies(product.state_id, drop_first=False).values,
   pd.get_dummies(product.store_id, drop_first=False).values,
   pd.get_dummies(product.cat_id, drop_first=False).values,
   pd.get_dummies(product.dept_id, drop_first=False).values,
   pd.get_dummies(product.state_id + product.cat_id, drop_first=False).values,
   pd.get_dummies(product.state_id + product.dept_id, drop_first=False).values,
   pd.get_dummies(product.store_id + product.cat_id, drop_first=False).values,
   pd.get_dummies(product.store_id + product.dept_id, drop_first=False).values,
   pd.get_dummies(product.item_id, drop_first=False).values,
   pd.get_dummies(product.state_id + product.item_id, drop_first=False).values,
   np.identity(NUM_ITEMS).astype(np.int8) #item :level 12
].T

weight_mat = weight_mat.astype("int8")
weight_mat, weight_mat.shape
weight_mat_csr = csr_matrix(weight_mat)
del weight_mat; gc.collect()
def cal_weight1(product):
    sales_train_val = sales
    d_name = ['d_' + str(i+1) for i in range(1913)]

    sales_train_val = weight_mat_csr * sales_train_val[d_name].values


    df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))

    start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1

    flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))<1

    sales_train_val = np.where(flag, np.nan, sales_train_val)

    # denominator of RMSSE / RMSSE
    weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1913-start_no)
    
    return weight1

weight1 = cal_weight1(product)
# Get the last 28 days for weight2
cols = ["d_{}".format(i) for i in range(1886, 1886+28)]

data = sales[["id", 'store_id', 'item_id'] + cols]

data = data.melt(id_vars=["id", 'store_id', 'item_id'], var_name="d", value_name="sale")
data = pd.merge(data, calendar, how = 'left', left_on = ['d'], right_on = ['d'])
data = data[["id", 'store_id', 'item_id', "sale", "wm_yr_wk"]]
data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')


def cal_weight2(data):
    # calculate the sales amount for each item/level
    df_tmp = data
    df_tmp['amount'] = df_tmp['sale'] * df_tmp['sell_price']
    df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum)
    df_tmp = df_tmp.values
    
    weight2 = weight_mat_csr * df_tmp 

    weight2 = weight2/np.sum(weight2)
    return weight2
    
weight2 = cal_weight2(data)
weight2.shape
def wrmsse(preds, y_true):
    # number of columns
    num_col = DAYS_PRED

    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
    
          
    train = weight_mat_csr*np.c_[reshaped_preds, reshaped_true]
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) / weight1) * weight2)
    
    return score
# WRMSSE score
DAYS_PRED = 10
dft = transform(df)
gtt = transform(gt)
wrmsse(dft["sale"].to_numpy(), gtt["sale"].to_numpy())
