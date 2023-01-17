import pandas as pd
from matplotlib import pyplot as plt

%matplotlib inline
TRAIN_FILE = "../input/train.csv"
TEST_FILE = "../input/test.csv"
f = 'AvSigVersion'
train_df = pd.read_csv(TRAIN_FILE, usecols=[f])
test_df = pd.read_csv(TEST_FILE, usecols=[f])
df = pd.concat([train_df,test_df])
unq_vals = df[f].value_counts().reset_index()
unq_vals = pd.concat([unq_vals, unq_vals['index'].str.replace('&#x17;','').str.split('.', expand=True).astype(int)], axis=1)
sorted_vals = unq_vals.sort_values([0,1,2,3]).reset_index(drop=True).reset_index()
vals_dict = sorted_vals.set_index('index')['level_0'].to_dict()
df[f] = df[f].map(vals_dict)
train_len = len(train_df)
train_df = df[:train_len]
test_df = df[train_len:]
_ = plt.hist([train_df[f],test_df[f]], stacked=False)
