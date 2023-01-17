import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
#データインポート

df_train = pd.read_csv('../input/train_ver2.csv')

df_test = pd.read_csv('../input/test_ver2.csv')

df_submission_sample = pd.read_csv('../input/sample_submission.csv')
#学習データ

df_train
#テストデータ

df_test
#提出データ

df_submission_sample.to_csv('test_submit.csv',index=False)
df_submission_sample