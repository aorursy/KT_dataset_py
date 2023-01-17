import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
%time train = pd.read_csv('../input/train.csv')
train.shape, train.columns
%time test = pd.read_csv('../input/test.csv')
test.shape, test.columns
train.dtypes.value_counts()
features = train.drop(columns=['ID', 'target'])
features.min().min(), features.max().max(), features.isnull().any().any()
test_features = test.drop(columns='ID').sample(n=features.shape[0], random_state=123)
test_features.min().min(), test_features.max().max(), test_features.isnull().any().any()
plt.figure(figsize=(5,5))
plt.spy((features > 0).values);
(features == 0).sum().sum() / features.size * 100
plt.figure(figsize=(5,5))
plt.spy((test_features > 0).values);
(test_features == 0).sum().sum() / test_features.size * 100
nunique = features.nunique()
no_info = nunique == 1
no_info.sum()
to_drop = nunique[no_info].index.values
train.drop(columns=to_drop, inplace=True)
features.drop(columns=to_drop, inplace=True)
test.drop(columns=to_drop, inplace=True)
test_features.drop(columns=to_drop, inplace=True)
train.loc[features.duplicated(keep=False), ['ID', 'target']]
trans = features.T
all_duplicates = trans[trans.duplicated(keep=False)].index
last_duplicates = trans[trans.duplicated()].index
all_duplicates, last_duplicates
test_sample = test_features.sample(n=features.shape[0], random_state=123)
trans_test = test_sample.T
trans_test[trans_test.duplicated(keep=False)].index
for i in range(len(all_duplicates)):
    for j in range(i + 1, len(all_duplicates)):
        col1, col2 = all_duplicates[i], all_duplicates[j]
        print(col1, col2, 'train:', sum(train[col1] != train[col2]), ' test:', sum(test_sample[col1] != test_sample[col2]))
train.target.describe()
fig, ax = plt.subplots()
plt.scatter(range(train.shape[0]), np.sort(train.target.values));
ax.set_yscale('log')
int_cols = features.columns[features.dtypes == np.int64].values
int_train = features[int_cols]
plt.figure(figsize=(5,10))
plt.spy((int_train > 0).values);
(int_train == 0).sum().sum() / int_train.size * 100
nunique_int = int_train.nunique()
fig, ax = plt.subplots()
nunique_int.hist(bins=300, bottom=0.1)
ax.set_xscale('log')
float_cols = features.columns[features.dtypes == np.float64].values
float_train = features[float_cols]
float_train = train[float_cols]
plt.figure(figsize=(5,10))
plt.spy((float_train > 0).values);
(float_train == 0).sum().sum() / float_train.size * 100
nunique_float = float_train.nunique()
fig, ax = plt.subplots()
nunique_float.hist(bins=300, bottom=0.1)
ax.set_xscale('log')
train.target = np.log1p(train.target)
%time train.to_feather('train.feather')
%time train = pd.read_feather('train.feather')
# No space left in Kaggle, but can be done locally
# %time test.to_feather('test.feather')
# %time test = pd.read_feather('test.feather')
