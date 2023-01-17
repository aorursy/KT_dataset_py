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
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype

import gc

import matplotlib.pyplot as plt
trainB=pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

trainB_meta=pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

testB=pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

trainw=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

testw=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

submission=pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")



    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df



trainB= reduce_mem_usage(trainB,use_float16=True)

testB= reduce_mem_usage(testB,use_float16=True)

trainw= reduce_mem_usage(trainw,use_float16=True)

testw= reduce_mem_usage(testw,use_float16=True)
#Concatenate weather data ,anargy consumption and building meta data for test and training sets

def merge(trainw,trainB,trainB_meta):

    trainB=trainB.merge(trainB_meta, left_on='building_id',right_on='building_id',how='left')

    train= trainB.merge(trainw,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

    del trainw

    del trainB

    del trainB_meta

    gc.collect()

    train = reduce_mem_usage(train,use_float16=True)

    return train;

train=merge(trainw,trainB,trainB_meta)

test=merge(testw,testB,trainB_meta)

train.columns
train.head(4)
train.shape
#The percentage of null data in each column in train set 

train.isnull().sum()*100/train.shape[0]
train.describe()
#The total of null data in each column in train set 

def miss_val(train):

    percent = (train.isnull().sum()).sort_values(ascending=False)

    percent.plot(kind="bar", figsize = (20,10), fontsize = 20)

    plt.xlabel("Columns", fontsize = 20)

    plt.ylabel("Number of rows", fontsize = 20)

    plt.title("Total Missing Values per column", fontsize = 20)

miss_val(train)
# The percentage of null lignes 

def miss_val_lines(train):



    print(len(np.where(train.isnull().sum(axis=1)==0)[0])*100/train.shape[0])

    print(len(np.where(train.isnull().sum(axis=1)==1)[0])*100/train.shape[0])

    print(len(np.where(train.isnull().sum(axis=1)==2)[0])*100/train.shape[0])

    print(len(np.where(train.isnull().sum(axis=1)==3)[0])*100/train.shape[0])

    print(len(np.where(train.isnull().sum(axis=1)==4)[0])*100/train.shape[0])

    print(len(np.where(train.isnull().sum(axis=1)==5)[0])*100/train.shape[0])

    print(len(np.where(train.isnull().sum(axis=1)==6)[0])*100/train.shape[0])

    print(len(np.where(train.isnull().sum(axis=1)==7)[0])*100/train.shape[0])

    print(len(np.where(train.isnull().sum(axis=1)==8)[0])*100/train.shape[0])

    print(len(np.where(train.isnull().sum(axis=1)==9)[0])*100/train.shape[0])

miss_val_lines(train)
train.dtypes
#The distribution of primary_use column

def cat_column(train):

    

    fig = plt.figure(figsize=(20,15))

    train.groupby('primary_use').primary_use.count().sort_values(ascending=False).plot.bar(ylim=0)

    plt.show()

cat_column(train)
# The change in energy consumption by date

import plotly.express as px

def energy_cons(train):   

    train['timestamp'] = pd.to_datetime(train.timestamp, format='%Y-%m-%d %H:%M:%S')

    train['date']=train['timestamp'].dt.date

    train_temp_df_meter = train.groupby('date')['meter_reading'].sum()

    train_temp_df_meter = train_temp_df_meter.to_frame().reset_index()

    fig = px.line(train_temp_df_meter, x='date', y='meter_reading')

    fig.show()

energy_cons(train)
test=test.drop('row_id',axis=1)
test.head(4)
test.shape
test.isnull().sum()*100/test.shape[0]
test.describe()
miss_val(test)
miss_val_lines(test)
cat_column(test)