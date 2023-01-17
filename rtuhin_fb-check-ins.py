# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('../input/train.csv')
#train_df['time'] = pd.to_datetime(train_df['time'],unit='s')
#print ('No. of unique places:', len(df['place_id'].unique()))

#print (df.describe())

# group by place id
grouped = df.groupby('place_id', as_index=False)

most_popular = grouped.size().sort_values(ascending=False)[:50]
print (most_popular)
df.plot(kind='hist', y = 'accuracy', bins=500)
