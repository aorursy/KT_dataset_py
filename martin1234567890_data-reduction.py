import pandas as pd 

import numpy as np 

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
def mem_usage(data):

    if isinstance(data, pd.DataFrame):

        usage_b = data.memory_usage(deep=True).sum()

    else: 

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_GB = usage_b / 1024 ** 2 / 1024

    return "{:03.2f} GB".format(usage_GB)
def reduce_memory(data):

    

    data_int = data.select_dtypes(include=['int'])

    converted_int = data_int.apply(pd.to_numeric,downcast='unsigned')

    

    data = pd.concat([data.drop(data_int.columns, axis=1),

                   converted_int],axis=1)

    

    obj_data = data.select_dtypes(include=['object'])



    for col in obj_data.columns:

        data[col] = data[col].astype('category')

        

    del data_int, converted_int, obj_data

        

    return data
print("train data", mem_usage(train))
train = reduce_memory(train)
print("train data", mem_usage(train))
one_percent = 8.14/100
print("{:03.2f}% reduction".format(4.41/one_percent))
train.to_csv("new_train.csv", index=False)
!ls -ls