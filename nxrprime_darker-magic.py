import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data1 = pd.read_csv('../input/m5-more-data-table-and-xgb/submission_lgbm.csv')

data2 = pd.read_csv('../input/m5-dark-magic/submission.csv')
for i in range(1,29):

    data1['F'+str(i)] *= 1.04
categories = []

for i in range(1, 29):

    categories.append(f'F{i}')
sub_col = data1['id']
all_cols = pd.DataFrame({})

all_cols['id'] = sub_col

all_cols[categories] = 0.60*data1[categories] + 0.40*data2[categories]
all_cols.to_csv('sub.csv', index=False)