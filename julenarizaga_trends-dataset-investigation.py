import pandas as pd

import numpy as np
directory = '/kaggle/input/trends-assessment-prediction/'

df_icn = pd.read_csv(directory + 'ICN_numbers.csv')

print(df_icn.shape)

print(df_icn['ICN_number'].unique().shape[0] == df_icn.shape[0])

print('The minimun ICN number:',  min(df_icn['ICN_number']))

print('The maximum ICN number:',  max(df_icn['ICN_number']))

df_icn.head(10)

# There are 53 unique ICN numbers. But they are not from 0 to 53, they are sparse
loading = pd.read_csv(directory + 'loading.csv')

print(loading.shape)

loading.columns

# it has 11,754 instances (rows) and 27 columns

loading.head(10)
reveal = pd.read_csv(directory + 'reveal_ID_site2.csv')

print(len(reveal['Id'].unique()))

reveal.head(10)

# This dataset has 510 unique id's
train = pd.read_csv(directory + 'train_scores.csv')

train.head(10)

# The training has the variables (5) for 5,877 unique id's
submission = pd.read_csv(directory + 'sample_submission.csv')

print('Number of unique ids in the test set:', len(submission)/5)

submission.head(10)