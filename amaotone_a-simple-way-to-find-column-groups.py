import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from collections import deque

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)
all_df = pd.concat([train, test]).reset_index(drop=True)
def get_leak_rows(df, col_groups):
    f1 = sum([cols[:-1] for cols in col_groups], [])
    f2 = sum([cols[1:] for cols in col_groups], [])

    d1 = df[f1].apply(tuple, axis=1).reset_index().rename(columns={0: 'key'})
    d2 = df[f2].apply(tuple, axis=1).reset_index().rename(columns={0: 'key'}) 
    d1.drop_duplicates(['key'], keep=False, inplace=True)
    d2.drop_duplicates(['key'], keep=False, inplace=True)
    return d1.merge(d2, how='inner', on='key')

def get_leak_cols(df, row_groups):
    f1 = sum([rows[:-1] for rows in row_groups], [])
    f2 = sum([rows[1:] for rows in row_groups], [])

    d1 = df.iloc[f1].apply(tuple, axis=0).reset_index().rename(columns={0: 'key'})
    d2 = df.iloc[f2].apply(tuple, axis=0).reset_index().rename(columns={0: 'key'}) 
    d1.drop_duplicates(['key'], keep=False, inplace=True)
    d2.drop_duplicates(['key'], keep=False, inplace=True)
    return d1.merge(d2, how='inner', on='key')

def get_groups(df):
    forward = df.set_index('index_x')['index_y'].to_dict()
    backward = df.set_index('index_y')['index_x'].to_dict()
    
    groups = []
    while len(forward.keys()):
        k, v = forward.popitem()
        backward.pop(v)
        k_first, v_first = k, v
        g = deque([k, v])
        while True:
            if v in forward:
                v = forward.pop(v)
                backward.pop(v)
                g.append(v)
            else:
                break
        while True:
            if k in backward:
                k = backward.pop(k)
                forward.pop(k)
                g.appendleft(k)
            else:
                break
        groups.append(list(g))
    return sorted(groups, key=len, reverse=True)
# initial col_groups
col_groups = [['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98']]
df = get_leak_rows(all_df, col_groups)
row_groups = get_groups(df)
row_groups = [r for r in row_groups if len(r) >= 5]  # select reliable row_groups

df = get_leak_cols(all_df, row_groups)
col_groups = get_groups(df)
print(f'find {len([c for c in col_groups if len(c) == 40])} col_groups')
with open('col_groups.txt', 'w') as f:
    for c in col_groups:
        if len(c) == 40:
            f.write(str(c) + '\n')