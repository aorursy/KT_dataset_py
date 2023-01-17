import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
hist_trans = pd.read_csv('../input/historical_transactions.csv')
new_trans = pd.read_csv('../input/new_merchant_transactions.csv')
hist_trans.shape, new_trans.shape
hist_trans.purchase_date = pd.to_datetime(hist_trans.purchase_date)
new_trans.purchase_date = pd.to_datetime(new_trans.purchase_date)
hist_trans.purchase_date.agg([np.min, np.max])
new_trans.purchase_date.agg([np.min, np.max])
hist_time_range = hist_trans.groupby('card_id', sort=False)['purchase_date'].agg([np.min, np.max])
new_time_range = new_trans.groupby('card_id', sort=False)['purchase_date'].agg([np.min, np.max])
hist_time_range.head()
new_time_range.head()
hist_time_range.shape, new_time_range.shape
set(hist_time_range.index.tolist()) - set(new_time_range.index.tolist())
set(new_time_range.index.tolist()) - set(hist_time_range.index.tolist())
time_range = hist_time_range.join(new_time_range, how='right', lsuffix='_hist', rsuffix='_new')
time_range.head()
time_range.shape
all(time_range.amax_hist < time_range.amin_new)
hist_month_num = (time_range.amax_hist - time_range.amin_hist).apply(lambda x: x.days / 30)
new_month_num = (time_range.amax_new - time_range.amin_new).apply(lambda x: x.days / 30)
plt.hist(hist_month_num.values, bins=50)
plt.title('Historical transactions time range')
plt.xlabel('Month')
plt.ylabel('Counts')
plt.hist(new_month_num.values, bins=50)
plt.title('New merchant transactions time range')
plt.xlabel('Month')
plt.ylabel('Counts')
hist_trans.dropna(axis=0, how='any', inplace=True)
new_trans.dropna(axis=0, how='any', inplace=True)
hist_trans.shape, new_trans.shape
hist_merchant_set = hist_trans.groupby('card_id')['merchant_id'].apply(lambda x: set(x.tolist()))
new_merchant_set = new_trans.groupby('card_id')['merchant_id'].apply(lambda x: set(x.tolist()))
hist_merchant_set.head()
merchant_set = pd.concat([hist_merchant_set, new_merchant_set], axis=1, join='inner')
merchant_set.columns = ['hist', 'new']
merchant_set.head()
intersection = merchant_set.apply(lambda x: len(set.intersection(x['hist'], x['new'])), axis=1)
intersection.head()
any(intersection.values)
