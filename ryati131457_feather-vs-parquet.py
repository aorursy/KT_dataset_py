import numpy as np 

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%%time

train = pd.read_feather('/kaggle/input/riiid-feather-files/train.feather')
train
train.to_parquet('tr2.parquet', compression=None)
%%time

t2 = pd.read_parquet('tr2.parquet')