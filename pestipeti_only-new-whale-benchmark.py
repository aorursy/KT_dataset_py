import pandas as pd
import numpy as np
train_df = pd.read_csv('../input/train.csv')
n_all = train_df.shape[0]
n_new_whale = train_df[train_df['Id'] == 'new_whale'].shape[0]
print("We have {}/{} ({:.2f}%) `new_whale` in the training set.".format(n_new_whale, n_all, n_new_whale/n_all*100))
res = []
for i in range(10):
    tmp_df = train_df.sample(frac=.2)
    n_all = tmp_df.shape[0]
    n_new_whale = tmp_df[tmp_df['Id'] == 'new_whale'].shape[0]
    res.append(n_new_whale/n_all)
    print("We have {}/{} ({:.2f}%) `new_whale` in the random subset #{}.".format(n_new_whale, n_all, n_new_whale/n_all*100, i))
    
print("Average: {:.4f}, std: {:.4f}".format(np.mean(res), np.std(res)))
test_df = pd.read_csv('../input/sample_submission.csv')
test_df.head()
test_df['Id'] = 'new_whale'
test_df.head()
n_all = test_df.shape[0]
# After submission (public LB score: 0.276)
n_new_whale = 2197
print("We have {}/{} ({:.2f}%) `new_whale` in the test set.".format(n_new_whale, n_all, n_new_whale/n_all*100))
test_df.to_csv('new_whale_benchmark.csv', index=False)