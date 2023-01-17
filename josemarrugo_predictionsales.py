import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
DATA_FOLDER = '../input/competitive-data-science-predict-future-sales/'
transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
transactions = pd.merge(transactions, items, on='item_id', how='left')
transactions = transactions.drop('item_name', axis=1)
transactions.head()
from itertools import product
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = []

for block_num in transactions['date_block_num'].unique():
    cur_shops = transactions.loc[transactions['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = transactions.loc[transactions['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))
    
grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype=np.int32)
grid.head()
grid.tail()
mean_transactions = transactions.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':'sum','item_price':np.mean}).reset_index()
mean_transactions = pd.merge(grid,mean_transactions,on=['date_block_num', 'shop_id', 'item_id'],how='left').fillna(0)
mean_transactions = pd.merge(mean_transactions, items, on='item_id',how='left')
mean_transactions.head()
mean_transactions.tail()
for type_id in ['item_id', 'shop_id', 'item_category_id']:
    for column_id, aggregator, aggtype in [('item_price',np.mean,'avg'),('item_cnt_day',np.sum,'sum'),('item_cnt_day',np.mean,'avg')]:
        
        mean_df = transactions.groupby([type_id,'date_block_num']).aggregate(aggregator).reset_index()[[column_id,type_id,'date_block_num']]
        mean_df.columns = [type_id+'_'+aggtype+'_'+column_id,type_id,'date_block_num']
        mean_transactions = pd.merge(mean_transactions, mean_df, on=['date_block_num',type_id], how='left')
mean_transactions.head(2)
mean_transactions.tail(2)
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:

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

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
mean_transactions = reduce_mem_usage(mean_transactions)
mean_transactions.head()
lag_variables  = list(mean_transactions.columns[7:])+['item_cnt_day']
lags = [1, 2, 3, 6]
from tqdm import notebook
for lag in notebook.tqdm(lags):

    sales_new_df = mean_transactions.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    mean_transactions = pd.merge(mean_transactions, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')
mean_transactions.head(3)
mean_transactions = mean_transactions[mean_transactions['date_block_num']>12]
for feat in mean_transactions.columns:
    if 'item_cnt' in feat:
        mean_transactions[feat]=mean_transactions[feat].fillna(0)
    elif 'item_price' in feat:
        mean_transactions[feat]=mean_transactions[feat].fillna(mean_transactions[feat].median())
cols_to_drop = lag_variables[:-1] + ['item_price', 'item_name'] # dropping all target variables but not "item_cnt_day" cause is target
training = mean_transactions.drop(cols_to_drop,axis=1)
import xgboost as xgb
xgbtrain = xgb.DMatrix(training.iloc[:, training.columns != 'item_cnt_day'].values, training.iloc[:, training.columns == 'item_cnt_day'].values)
param = {'max_depth':10, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':0.3, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'} # random parameters
bst = xgb.train(param, xgbtrain)
filename='xgboost_model.sav'
pickle.dump(bst, open(filename, 'wb'))
loaded_bst = pickle.load(open(filename, 'rb'))
x=xgb.plot_importance(bst)
x.figure.set_size_inches(10, 30) 
cols = list(training.columns)
del cols[cols.index('item_cnt_day')] # eliminate target feature col name
[cols[x] for x in [2, 0, 5, 8, 4, 1, 3, 9, 33]]
training.columns
test            = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
test.head()
test['date_block_num'] = 34
test = pd.merge(test, items, on='item_id', how='left')
from tqdm import notebook
for lag in notebook.tqdm(lags):

    sales_new_df = mean_transactions.copy()
    sales_new_df.date_block_num += lag
    sales_new_df = sales_new_df[['date_block_num','shop_id','item_id']+lag_variables]
    sales_new_df.columns = ['date_block_num','shop_id','item_id']+ [lag_feat+'_lag_'+str(lag) for lag_feat in lag_variables]
    test = pd.merge(test, sales_new_df,on=['date_block_num','shop_id','item_id'] ,how='left')
test.head()
_test = set(test.drop(['ID', 'item_name'], axis=1).columns)
_training = set(training.drop('item_cnt_day',axis=1).columns)
for i in _test:
    assert i in _training
for i in _training:
    assert i in _test
assert _training == _test
test = test.drop(['ID', 'item_name'], axis=1)
for feat in test.columns:
    if 'item_cnt' in feat:
        test[feat]=test[feat].fillna(0)
    elif 'item_price' in feat:
        test[feat]=test[feat].fillna(test[feat].median())
test[['shop_id','item_id']+['item_cnt_day_lag_'+str(x) for x in [1,2,3]]].head()
print(training[training['shop_id'] == 5][training['item_id'] == 5037][training['date_block_num'] == 33]['item_cnt_day'])
print(training[training['shop_id'] == 5][training['item_id'] == 5037][training['date_block_num'] == 32]['item_cnt_day'])
print(training[training['shop_id'] == 5][training['item_id'] == 5037][training['date_block_num'] == 31]['item_cnt_day'])
xgbpredict = xgb.DMatrix(test.values)
pred = bst.predict(xgbpredict)
pd.Series(pred).describe()
pred = pred.clip(0, 20)
pred.sum()
pd.Series(pred).describe()
sub_df = pd.DataFrame({'ID':test.index,'item_cnt_month': pred })
sub_df.head()
sub_df.to_csv('submission.csv',index=False)
