import numpy as np

import pandas as pd 



train_raw = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
train_raw.info()
old_memory_usage = train_raw.memory_usage(deep = True)
ordinality_of_cats = train_raw.describe(include = [np.object]).T.sort_values('unique')

ordinality_of_cats
train_less_memory = train_raw.copy()

low_card_cols = ordinality_of_cats.query('unique < 300').index.tolist()

for col in low_card_cols:

    train_less_memory[col] = train_raw[col].astype('category')
train_less_memory.dtypes
train_less_memory.memory_usage(deep = True)/old_memory_usage