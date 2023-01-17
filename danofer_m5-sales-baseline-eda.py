import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

print(f"train shape {train_sales.shape}")

submission_file = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')

print(f"submission_file shape {submission_file.shape}")
days = range(1, 1913 + 1)

time_series_columns = [f'd_{i}' for i in days]



time_series_data = train_sales[time_series_columns]
print(train_sales.columns)

id_df_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
train_sales[id_df_columns].nunique()
train_sales[id_df_columns + time_series_columns].head(2)

##opt - drop out test set rows

# train_sales = train_sales[id_df_columns + time_series_columns]
# https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtypes

        

        if col_type != object:

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

        #else: df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
# ## reduce memory usage. There's an imporved version of this function that also saves data as categoircals type, but that can affect joins if not handled explicitly

# train_sales = reduce_mem_usage(train_sales)



### we know the max range of the sales cols, let's just set them all to int 16 (some are int8 , but that doesn't matter if we 'll cast it)

display(train_sales.info())

train_sales[time_series_columns] = train_sales[time_series_columns].astype(np.int16)
# train_sales[id_df_columns] = train_sales[id_df_columns].astype('category')

display(train_sales.info())
train_sales.dtypes
train_sales
submission_file
time_series_data
calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv",parse_dates=["date"])

print(calendar.shape)

prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

print(prices.shape)
calendar.tail(3)
#no need to keep the textual weekday name, we have it from wday + data. Saturday = 1,Sunday	2, Friday	7

calendar.drop("weekday",axis=1,inplace=True)



## we  drop the prefix from the calendar date/d column for easy merging with sales data. .

calendar["d"] = calendar["d"].replace("d_","",regex=True).astype(int)

calendar
prices
print(f"After reshaping to 1 row per id per day/date, we would have: {train_sales.shape[0]*time_series_data.shape[1]} rows")

## 58 million rows. many sparse likely
%%time

pd.wide_to_long(train_sales.head(3),"d_",i=id_df_columns,j="sales").reset_index()
stores_list = list(set(train_sales["store_id"]))

stores_list
%%time

### reshape incrementally - hopefully this will help with memory errors

dfs= []

for st in stores_list:  

    df = train_sales.loc[train_sales["store_id"]==st]#.head()

    dfs.append(pd.wide_to_long(df,"d_",i=id_df_columns,j="day").reset_index())

    

df = pd.concat(dfs)

df.rename(columns={"d_":"sales"})

del(dfs)

print(df.shape)

df
df.tail()
# %%time

# train_sales = pd.wide_to_long(train_sales,"d_",i=id_df_columns,j="sales").reset_index()

# print(train_sales.shape)

# train_sales
df.to_csv("sales_basic_v1_all.csv.gz",index=False,compression="gzip")
validation_ids = train_sales['id'].values

evaluation_ids = [i.replace('validation', 'evaluation') for i in validation_ids]

ids = np.concatenate([validation_ids, evaluation_ids])

predictions = pd.DataFrame(ids, columns=['id'])

forecast = pd.concat([forecast] * 2).reset_index(drop=True)

predictions = pd.concat([predictions, forecast], axis=1)

predictions.to_csv('submission.csv', index=False)