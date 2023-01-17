# load library

import numpy as np

import pandas as pd
# read train_features.csv

df_train_f = pd.read_csv("../input/lish-moa/train_features.csv", header=0)



# show detail

print('Number of rows : ', df_train_f.shape[0])

print('Number of cols : ', df_train_f.shape[1])

print('cp_type : ', df_train_f['cp_type'].drop_duplicates().values)

print('cp_dose : ', df_train_f['cp_dose'].drop_duplicates().values)



df_train_f.head()
# read test_features.csv

df_test_f = pd.read_csv("../input/lish-moa/test_features.csv", header=0)



# show detail

print('Number of rows : ', df_test_f.shape[0])

print('Number of cols : ', df_test_f.shape[1])

print('cp_type : ', df_test_f['cp_type'].drop_duplicates().values)

print('cp_dose : ', df_test_f['cp_dose'].drop_duplicates().values)



df_test_f.head()
# read train_targets_nonscored.csv

df_train_tn = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv", header=0)



# show detail

print('Number of rows : ', df_train_tn.shape[0])

print('Number of cols : ', df_train_tn.shape[1])

df_train_tn.head()
# read train_targets_scored.csv

df_train_ts = pd.read_csv("../input/lish-moa/train_targets_scored.csv", header=0)



# show detail

print('Number of rows : ', df_train_ts.shape[0])

print('Number of cols : ', df_train_ts.shape[1])

df_train_ts.head()
# read sample_submission.csv

df_sub = pd.read_csv("../input/lish-moa/sample_submission.csv", header=0)



# show detail

print('Number of rows : ', df_sub.shape[0])

print('Number of cols : ', df_sub.shape[1])

df_sub.head()
df_sub.to_csv('submission.csv', index=False)