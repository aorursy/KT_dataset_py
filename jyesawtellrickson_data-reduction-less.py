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
print(f'Starting train data is {mem_usage(train)}')
train_int = train.select_dtypes(include=['int'])

converted_int = train_int.apply(pd.to_numeric, downcast='unsigned')
print(f'Previous int data is {mem_usage(train_int)}')

print(f'New int data is      {mem_usage(converted_int)}')
train = pd.concat([train.drop(train_int.columns, axis=1),

                   converted_int], axis=1)
train.dtypes
for cat in train.select_dtypes('object').columns:

    print(f'Column: {cat}')

    print(f'Before: {mem_usage(train[cat].to_frame())}')

    print(f'After: {mem_usage(train[cat].astype("category").to_frame())}')
for col in [c for c in train.select_dtypes(include=['object']).columns if c not in ('event_data', 'timestamp')]:

    train[col] = train[col].astype('category')
print(f'Final size {mem_usage(train)}')
print("{:03.2f}% reduction".format((3.81/8.14)*100))