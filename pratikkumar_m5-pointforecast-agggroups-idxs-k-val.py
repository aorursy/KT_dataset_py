# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pickle, sys

from collections import defaultdict
sales_train_val = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
sales_train_val

with open('../input/m5-point-forecast-valweights/weights_df', 'rb') as f:

    val_weights = pickle.load(f)
val_weights
mask = (val_weights.weight != 0).values
val_weights_mask = val_weights.loc[mask]
val_weights_mask = val_weights_mask.reset_index(drop=True)
# item_store and all_id implicit

agg_levels = (

            'cat_id',

            'state_id',

            'dept_id',

            'store_id',

            'item_id',

            ['state_id', 'cat_id'],

            ['state_id', 'dept_id'],

            ['store_id', 'cat_id'],

            ['store_id', 'dept_id'],

            ['item_id', 'state_id'],

            )
sales_train_val_mask = sales_train_val.loc[mask]
sales_train_val_mask = sales_train_val_mask.reset_index(drop=True)
agg_levels
agg_groups = []

for ag in agg_levels:

    st_gp = sales_train_val_mask.groupby(by=ag)

    for _,_df in st_gp:

        idxs = _df.index.values

        ser = _df.iloc[:,6:].values.sum(axis=0)

        nz_i = np.sort(np.where(ser>0)[0])[0]

        K = ((ser[nz_i+1:] - ser[nz_i:-1])**2).mean()

        K = (1.0 / K)**0.5

        agg_groups.append((idxs,K))
len(agg_groups)
all_ser = sales_train_val_mask.iloc[:,6:].values.sum(axis=1)

all_nz_i = np.sort(np.where(all_ser>0)[0])[0]

all_K = ((all_ser[all_nz_i+1:] - all_ser[all_nz_i:-1])**2).mean()

all_K = (1.0 / all_K)**0.5
all_idxs = sales_train_val_mask.index.values
agg_groups.append((all_idxs, all_K))
with open('agg_groups_idxs_K','wb') as f:

    pickle.dump(agg_groups, f)
! ls -alrh