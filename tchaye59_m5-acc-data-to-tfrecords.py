# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename)) 



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')
import os

def log(*args):

    os.system(f'echo \"{args}\"')

    print(*args)
! pip install tensorflow_datasets
import tensorflow as tf

import tensorflow_datasets.public_api as tfds

from sklearn.preprocessing import LabelEncoder

import gc

import pickle

import pandas as pd

import numpy as np
df_stve = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

df_cal = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

df_price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
# We keep our encoders in a dictionary in case we need it later

encoders = {}
# sales from this first day to 1941st day are represented by columns d_1 to d_1941  with a row representing an items in a store

df_stve.head(2)
# We will not use the id column

df_stve.drop(['id','dept_id','state_id'],axis=1,inplace=True)
# No missing values

df_stve.isna().sum().sum()
# Now we encode categorical features to numerical values

for col in df_stve.columns:

    if not np.issubdtype(df_stve[col].dtype, np.number):

        if col not in encoders:

            encoders[col] = LabelEncoder().fit(df_stve[col])

        df_stve[col] = encoders[col].transform(df_stve[col])

df_stve.head(2)
df_cal.head(2)
print(df_cal.shape)

df_cal.isna().sum()
# event columns are missing we will fill them with NA

cols = ['event_name_1','event_type_1'] # We will after drop event_name_2 and event_type_2: on 1969 records 1964 are null

df_cal[cols] = df_cal[cols].fillna('NA')
df_cal.head(2)
# We don't want to include these columns in our final dataset so we drop them

df_cal.drop(['date','weekday','event_name_2', 'event_type_2','snap_CA','snap_TX','snap_WI'],axis=1, inplace=True)

# We remove  the d_ prefix in d column

df_cal["d"] = df_cal["d"].apply(lambda x : int(x.split("_")[-1])).astype(int)
# encoding 

for col in df_cal.columns:

    log(col,df_cal[col].dtype)

    if not np.issubdtype(df_cal[col].dtype, np.number):

        if col not in encoders:

            encoders[col] = LabelEncoder().fit(df_cal[col])

        df_cal[col] = encoders[col].transform(df_cal[col]).astype(np.int32)



df_cal.head()
df_price.head(2)
df_price.isna().sum() # No missing
# encoding 

for col in df_price.columns:

    log(col,df_price[col].dtype)

    if not np.issubdtype(df_price[col].dtype, np.number):

        if col not in encoders:

            encoders[col] = LabelEncoder().fit(df_price[col])

        df_price[col] = encoders[col].transform(df_price[col]).astype(np.int32)
# Thanks for the guy who wrote this function 😂️.

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

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: log('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
df_stve = df_stve.astype('int16')

df_cal = reduce_mem_usage(df_cal)

df_price = reduce_mem_usage(df_price)

gc.collect()
df_price.head(1)
df_cal.head(1) # We can merge them on wm_yr_wk
%%time

df_cal_price = df_cal.merge(df_price,on="wm_yr_wk")

df_cal_price.sort_values(['store_id','item_id','d'],inplace=True);

# For this challenge, memory is precious so we never miss an opportunity to collect garbage

del df_cal   

del df_price

gc.collect()
# Saturday and Sunday are respectively encoded 1,2

df_cal_price['weekend']= df_cal_price['wday'].apply(lambda x: 1 if x ==1 or x==2 else 0)

df_cal_price.head()
# Mean

def sell_price_rolling_mean(item_data):

    windows = [7,14,28]

    for rolling in windows:

        item_data[f'spma-{rolling}'] = item_data["sell_price"].rolling(window=rolling,min_periods=1).mean()

        item_data[f'spma-{rolling}'].fillna(0.,inplace=True)

    return item_data



#MIN

def sell_price_rolling_min(item_data):

    windows = [28]

    for rolling in windows:

        item_data[f'spmin-{rolling}'] = item_data["sell_price"].rolling(window=rolling,min_periods=1).min()

        item_data[f'spmin-{rolling}'].fillna(0.,inplace=True)

    return item_data



#MAX

def sell_price_rolling_max(item_data):

    windows = [28]

    for rolling in windows:

        item_data[f'spmax-{rolling}'] = item_data["sell_price"].rolling(window=rolling,min_periods=1).max()

        item_data[f'spmax-{rolling}'].fillna(0.,inplace=True)

    return item_data
df_cal_price = df_cal_price[df_cal_price['d'] <= 1941] #We only have 1941 days sales data 
df_cal_price_group = df_cal_price.groupby(['item_id','store_id'])

# Now we are ready to build our final dataset 😎️
# The columns in our final dataset

columns = ['item_id','cat_id','store_id','n_sales',] # These are from the sales df

# We will build these later (means and stds of sales)

columns += ['smean14', 'smean28', 'ssum28', 'smax28', 'smin28']

# calendar and price data

columns += ['wday', 'month', 'year', 'd', 'event_name_1','event_type_1', 'sell_price', 'weekend','spma-7', 'spma-14', 'spma-28']



len(columns)
# We save metadata

pickle.dump(columns, open("columns.pkl","wb"))

pickle.dump(encoders, open("encoders.pkl","wb"))

gc.collect()
class MyFullDataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.1.0')

    

    def _split_generators(self, dl_manager):

        return [

            tfds.core.SplitGenerator(

                    name=f'train',

                    gen_kwargs={

                    },

            )

        ]

    

    def _info(self):

        # Each record's feature matrice shape will be like (days, features)

        shape = (1941,len(columns))

        return tfds.core.DatasetInfo(

            builder=self,

            description=(""),

            features=tfds.features.FeaturesDict({

                "input": tfds.features.Tensor(shape=shape,dtype=tf.float32),

                "key": tfds.features.Tensor(shape=(),dtype=tf.int32),

            }),

        )

    

    def _generate_examples(self,**args):

        days = np.arange(1,1942)

        # We yield the time series one by one

        for i in range(df_stve.shape[0]):

            data = df_stve.iloc[i]

            item_id = data['item_id']

            store_id= data['store_id']

            data = data.values

            # repeat the first 5 cols for all days

            ids = np.repeat(data[:3].reshape(1,-1),1941,axis=0)

            

            # Feature engineering: we can brig some previous sales signals in present 

            df_sales = pd.DataFrame({'n_sales':data[3:]})

            

            df_sales['smean7'] = df_sales['n_sales'].rolling(window=7,min_periods=1).mean()

            df_sales['smean28'] = df_sales['n_sales'].rolling(window=28,min_periods=1).mean()

            

            df_sales['ssum28'] = df_sales['n_sales'].rolling(window=28,min_periods=1).sum()

            

            df_sales['smax28'] = df_sales['n_sales'].rolling(window=28,min_periods=1).max()

            df_sales['smin28'] = df_sales['n_sales'].rolling(window=28,min_periods=1).min()

            

            df_sales.fillna(0,inplace=True)

            #print(df_sales.columns)

            

            # We get calendar and price data

            cal_data = df_cal_price_group.get_group((item_id,store_id))

            # There is no price and calendar data for some days so we have to check and fill with -1

            if cal_data.shape[0] != 1941:

                missing_days = np.setxor1d(days, cal_data['d'].values)

                cal_data = cal_data.append(pd.DataFrame({'d':missing_days}),ignore_index=True)

            cal_data.sort_values(['d'],inplace=True);

            cal_data.fillna(-1,inplace=True)

            

            cal_data = sell_price_rolling_mean(cal_data)

            #cal_data = sell_price_rolling_max(cal_data)

            #cal_data = sell_price_rolling_min(cal_data)

            #print(cal_data.columns)

            cal_data = cal_data[['wday', 'month', 'year', 'd', 'event_name_1','event_type_1', 'sell_price', 'weekend','spma-7', 'spma-14', 'spma-28']]

            

            

            # We combine everything in one feature matrice with a shape like (days, features)

            # The key is the record position and will be useful later to generate the test samples

            input_ = np.c_[ids,df_sales.values,cal_data].astype(np.float32)

            yield i, {

                'input':input_,

                'key':i,

            }
%%time

### Create a local copy 



data_dir='./' 

builder = MyFullDataset(data_dir=data_dir)

# The following line creates the dataset folder containing the tf records files in /kaggle/input

builder.download_and_prepare() 
%%time

### Copy to GS Bucket

# To not worry about permission issues the bucket is publicly available in both reads and writes but will be deleted at the end of the competition

gs_path = 'gs://bucket59'

builder = MyFullDataset(data_dir=gs_path)

# If you opted for a bucket, change the gs_path to your bucket's path and uncomment this line

#builder.download_and_prepare()