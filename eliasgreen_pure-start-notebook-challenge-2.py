import numpy as np 

import pandas as pd

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

ssub = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
train.head(2)
train.info()
train.describe()
test.head(2)
test.info()
test.describe()
combined_df = train.append(test, ignore_index=True, sort=False)

combined_df.info()