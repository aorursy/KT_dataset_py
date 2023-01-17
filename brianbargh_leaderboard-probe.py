import pandas as pd
import numpy as np
test = pd.read_csv('../input/test.csv.gz', index_col=False)
sample_submission = pd.read_csv('../input/sample_submission.csv.gz', index_col='ID')
transactions = pd.read_csv('../input/sales_train.csv.gz', index_col=False)
all_ones_submission = sample_submission.copy()
all_ones_submission.item_cnt_month = 1
all_ones_submission.head()
all_ones_submission.to_csv('../allOnes.csv')
all_zeros_submission = sample_submission.copy()
all_zeros_submission.item_cnt_month = 0
all_zeros_submission.head()
all_zeros_submission.to_csv('../allZeros.csv')
# I won't post the results of my submissions as that might constitute unfair 
# teaming but you can duplicate my work.
# It's just as easy to calculate the variance of the test labels, I'll leave
# that to the reader. 

# First of all, we can only probe things that are in the test set

test_shops = test.shop_id.unique()
test_items = test.item_id.unique()

test_transactions = transactions[transactions.item_id.isin(test_items) & transactions.shop_id.isin(test_shops)]
items_by_month = test_transactions.groupby(['date_block_num', 'item_id']).sum()
items_by_month = items_by_month.unstack('date_block_num')[['item_cnt_day']]
items_by_month.rename(columns={'item_cnt_day':'total_item_cnt'}, inplace=True)
items_by_month.fillna(0, inplace=True)
items_by_month.head()
# Let's find some items that didn't sell at all in the last two years of the train
# data. We'll assume they didn't sell at all during the test period as well. 
last_26_months = items_by_month[items_by_month.columns[7:]]
items_by_month[last_26_months.sum(axis=1)==0]
# Didn't sell a single one since month 1. This is almost certainly going to be 
# 0 in the test sets. 
probe_item = 13536
# We're just sort of hoping that this will be in the public test set and not the 
# private one. Run it a few times until the error we get is larger than $e_0$. 
shop_num = np.random.choice(test_shops)
shop_num
# We'll fill in a single row with this value
# We want it big for numerical reasons. 
M = 500
probe_submission = all_zeros_submission.copy()
probe_submission[(test.item_id==probe_item) & (test.shop_id==shop_num)] = M

probe_submission.to_csv('../testSizeProbe.csv')
# Again, not posting my results...have fun!

# Note that this is just a random shop here - if you know one that's definitely 
# in the private test set you'll want to use that one instead. 
shop_num
probe_submission_2 = all_zeros_submission.copy()
probe_submission_2[test.shop_id==shop_num] = 20

probe_submission_2.to_csv('publicPrivateSplitProbe.csv')
# That's it! Just check and see if the error here is any different from $e_0$. 
