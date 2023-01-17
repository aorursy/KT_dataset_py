# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
historical_tx_df = pd.read_csv('../input/historical_transactions.csv')
historical_tx_df.info(memory_usage='deep')
for dtype in ['float','int','object']:
    selected_dtype = historical_tx_df.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
'''
Function to count unique values in the columsn. 

Arguments:
df_columns: List of columns in `his_int` dataframe
'''

def count_unique( df_columns):
    for i in df_columns:
        print("Number of unique vals in",i,"are",len(his_int[i].unique()))
his_int = historical_tx_df.select_dtypes(include=['integer'])
count_unique(his_int.columns)
converted_obj = pd.DataFrame()

for col in his_int.columns:
    num_unique_values = len(his_int[col].unique())
    num_total_values = len(his_int[col])
    if num_unique_values / num_total_values < 0.5:
        converted_obj.loc[:,col] = his_int[col].astype('category')
    else:
        converted_obj.loc[:,col] = his_int[col]
print(mem_usage(his_int))
print(mem_usage(converted_obj))
converted_obj.info()
optimized_hist_df = pd.DataFrame()
optimized_hist_df[converted_obj.columns] = converted_obj
mem_usage(optimized_hist_df)
his_int = historical_tx_df.select_dtypes(include=['float'])
his_int_na = his_int.category_2.fillna(6)
conv_obj = his_int_na.astype('int')
mem_usage(conv_obj)
conv_obj2 = conv_obj.astype('category')
mem_usage(conv_obj2)
his_int["category_2"] = conv_obj2
optimized_hist_df[his_int.columns] = his_int
mem_usage(optimized_hist_df)

his_int = historical_tx_df.select_dtypes(include=['object'])
his_int.head()
count_unique(his_int.columns)
converted_obj = pd.DataFrame()

for col in his_int.columns:
    num_unique_values = len(his_int[col].unique())
    num_total_values = len(his_int[col])
    if num_unique_values/num_total_values < 0.33:
        converted_obj.loc[:,col] = his_int[col].astype('category')
    else:
        converted_obj.loc[:,col] = his_int[col]
print(mem_usage(his_int))
print(mem_usage(converted_obj))
optimized_hist_df[converted_obj.columns] = converted_obj
mem_usage(optimized_hist_df)
optimized_hist_df['purchase_date'] =  pd.to_datetime(historical_tx_df.purchase_date, format='%Y%m%d %H:%M:%S')
mem_usage(optimized_hist_df)
optimized_hist_df.head()
per_df_red = (float(mem_usage(optimized_hist_df).split()[0]) / float(mem_usage(historical_tx_df).split()[0]) )*100
per_df_red
