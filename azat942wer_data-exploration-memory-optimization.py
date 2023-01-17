import pandas as pd
import os
import numpy as np 
import plotly.express as px
import gc
import warnings

warnings.filterwarnings("ignore")
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
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
def read_data(directory=r'../input/m5-forecasting-accuracy/'):
    print('reading calendar')
    cal = pd.read_csv(directory+r'calendar.csv')    
    cal = reduce_mem_usage(cal)
    
    print('reading sales_train_validation')
    sales_train_val = pd.read_csv(directory+r'sales_train_validation.csv')
    sales_train_val = reduce_mem_usage(sales_train_val)
    
    print('reading sample_submission')
    sample_submission = pd.read_csv(directory+r'sample_submission.csv')
    sample_submission = reduce_mem_usage(sample_submission)
    
    print('reading sell_prices')
    sell_prices = pd.read_csv(directory+r'sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)

    return cal, sales_train_val, sample_submission, sell_prices
# requires some time to load all data
cal, sales_train_val, sample_submission, sell_prices  = read_data()
cal.head(3)
sales_train_val.head(3)
sample_submission.head(3)
sell_prices.head(3)
# melting the dataset
print('melting')
sales_train_val_melted = pd.melt(sales_train_val, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='day', value_name='demand')
sales_train_val_melted = sales_train_val_melted.drop_duplicates()
del sales_train_val

sales_train_val_melted.head(3)
# lets group data by store_id and calculate total demand for each day by  each store
test = sales_train_val_melted.groupby(['store_id', 'day']).agg({'demand':'sum'})
test['store_id'] = [x[0] for x in test['demand'].index.tolist()]
test['day'] = [x[1] for x in test['demand'].index.tolist()]
test.reset_index(drop=True, inplace=True)
test['day_number'] = test['day'].apply(lambda x: int(x.split('_')[1]))
test = test.sort_values(by='day_number')
# test.head(3)

# DEMAND VISUALIZATION FOR EACH STORE 
fig = px.line(test,x='day_number', y="demand", color='store_id')
fig.show()

del test
gc.collect()
from sklearn.preprocessing import LabelEncoder

# lets remove wday, month, year
cal = cal[['date', 'wm_yr_wk', 'd', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']]

cal['event_name_1'] = cal['event_name_1'].fillna('No data')
cal['event_type_1'] = cal['event_type_1'].fillna('No data')
cal['event_name_2'] = cal['event_name_2'].fillna('No data')
cal['event_type_2'] = cal['event_type_2'].fillna('No data')


# LABEL ENCODING, ENCODING event_name_1, event_type_1, event_name_2, event_type_2
label_encoder = LabelEncoder()
cal['event_name_1'] = label_encoder.fit_transform(cal['event_name_1'])
cal['event_type_1'] = label_encoder.fit_transform(cal['event_type_1'])
cal['event_name_2'] = label_encoder.fit_transform(cal['event_name_2'])
cal['event_type_2'] = label_encoder.fit_transform(cal['event_type_2'])

# converting 
cal = reduce_mem_usage(cal)
cal.head(3)
# LABEL ENCODING, ENCODING event_name_1, event_type_1, event_name_2, event_type_2
label_encoder = LabelEncoder()
sales_train_val_melted['dept_id'] = label_encoder.fit_transform(sales_train_val_melted['dept_id'])
sales_train_val_melted['cat_id'] = label_encoder.fit_transform(sales_train_val_melted['cat_id'])
sales_train_val_melted['state_id'] = label_encoder.fit_transform(sales_train_val_melted['state_id'])


# LETS ADD:
# 1) ID_CONVERTER & 2) ITEM_ID_CONVERTER 
id_converter = LabelEncoder()
sales_train_val_melted['id'] = id_converter.fit_transform(sales_train_val_melted['id'])

item_id_converter = LabelEncoder()
sales_train_val_melted['item_id'] = item_id_converter.fit_transform(sales_train_val_melted['item_id'])

# converting dept_id, cat_id, state_id to another type
sales_train_val_melted = reduce_mem_usage(sales_train_val_melted)

sales_train_val_melted.head(3)
# encode item_id column
sell_prices['item_id'] = item_id_converter.transform(sell_prices['item_id'])

sell_prices = reduce_mem_usage(sell_prices)

# remove not_significant values after the dot 
sell_prices['sell_price'] = sell_prices['sell_price'].apply(lambda x: round(float(x),3))

sell_prices.head(3)

