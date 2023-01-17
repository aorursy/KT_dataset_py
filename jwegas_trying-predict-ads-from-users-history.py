import numpy as np

import pandas as pd
train = pd.read_csv('../input/clicks_train.csv', nrows=1000000)
train.info()
events = pd.read_csv('../input/events.csv', usecols=['display_id', 'uuid'])
events.info()
events = events.merge(train, how='inner', on='display_id')
events.head()
tmp = events[events['clicked']==1].groupby('uuid')['ad_id'].apply(list).reset_index()
tmp[tmp['ad_id'].apply(lambda x: len(x) > 1)]