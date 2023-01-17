import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn
pd.set_option('display.height', 1000)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
train, test, macro = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv'), pd.read_csv('../input/macro.csv')
test['price_doc'] = 0

df = pd.concat([train,test], ignore_index=True)
df.columns
len(train), len(test)
# in MB

df.memory_usage().sum()/1024/1024
df.describe()
df[df['price_doc']>0]['price_doc'].hist(bins=100)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
df.groupby(df['price_doc']>0)['timestamp'].hist()