import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

del train['ID'], test['ID'], train['target']
all_data = train.append(test)
all_data.reset_index(drop=True, inplace=True)

cols_with_onlyone_val = train.columns[train.nunique() == 1]

# Uncomment for fun:
#all_data.drop(cols_with_onlyone_val, axis=1, inplace=True)  
# GOAL: If user's col=min, and that min value only appears once in the user,
# then for certain, that col is NOT an aggregate
def notagg(row):
    row_nz = row[row>0]
    if row_nz.shape[0] == 0: return row # row is all 0s, so we return false=0 that it's not an agg row
    
    min_nz = row_nz.min()
    check  = (row_nz==min_nz).sum()
    
    # Min value occurs more than once, we can't learn anything about this column (min val column);
    # as such, we can't learn anything about this row
    if check>1:
        row = 0
        return row
    
    # Otherwise, min-val only occurs once! That col is NOT an aggregate
    return (row==min_nz).astype(np.int)  # only min-col will be marked=1
# Apply the above function to all rows:
cols_not_agg = all_data.apply(notagg, axis=1)
cols_not_agg.shape
# Cool, now look at each column and see if that column ever gets disqualified
cols_not_agg = cols_not_agg.max(axis=0)
cols_not_agg.shape # Make sure we're looking @ columns
cols_not_agg.sum()
which = cols_not_agg[cols_not_agg==0].index.tolist()
which
# Start with the easiest canidates. Let's see which of these columns has the least number of non-0 values
check = train[which]
check = check>0
pd.concat([check.sum(axis=0).sort_values(), 100 * check.sum(axis=0).sort_values() / train.shape[0]], axis=1)
nz = (all_data>0).sum(axis=0)
nz = nz.sort_values(ascending=False)
nz.shape
plt.plot(nz.values/all_data.shape[0])
plt.show()
