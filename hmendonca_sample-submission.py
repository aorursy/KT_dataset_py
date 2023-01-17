import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
!ls ../input
df_train = pd.read_csv('../input/train.csv')

print(df_train.describe())

df_train.head()
_ = df_train.sirna.hist()
# most common values

value_counts = df_train.sirna.value_counts()

value_counts.iloc[:6]
ss = pd.read_csv('../input/sample_submission.csv')

ss.sirna = int(df_train.sirna.median())

ss.head()
ss.to_csv('sample_submission.csv', index=False)