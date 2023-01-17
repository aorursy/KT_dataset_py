import pandas as pd
import matplotlib.pyplot as plt 
train = pd.read_csv('../input/train.csv'); train = train[train.columns[2:]]; train_row_nzr = train.astype(bool).sum(axis=1) / train.shape[1]; train_col_nzr = train.astype(bool).sum(axis=0) / train.shape[0]
test = pd.read_csv('../input/test.csv'); test = test[test.columns[1:]]; test_row_nzr = test.astype(bool).sum(axis=1) / test.shape[1]; test_col_nzr = test.astype(bool).sum(axis=0) / test.shape[0];

train.to_sparse(0.).density, test.to_sparse(0.).density
fig, axarr = plt.subplots(1,2, figsize=(15, 6))
fig.suptitle('Ratio of non-zeros per row')
train_row_nzr.plot.hist(bins=100, ax=axarr[0], title='train.csv')
test_row_nzr.plot.hist(bins=100, ax=axarr[1], title='test.csv')

train_row_nzr.max(), test_row_nzr.max()
fig, axarr = plt.subplots(1,2, figsize=(15, 6))
fig.suptitle('Ratio of non-zeros per column')
train_col_nzr.plot.hist(bins=100, ax=axarr[0], title='train.csv')
test_col_nzr.plot.hist(bins=100, ax=axarr[1], title='test.csv')

train_col_nzr.max(), test_col_nzr.max()
