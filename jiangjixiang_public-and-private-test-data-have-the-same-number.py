import numpy as np 

import pandas as pd

import os

import time



print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

print(os.listdir('../input'))



train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print(train_df.shape)

print(test_df.shape)
sub_df = pd.read_csv('../input/otheraptossubmission/submission.csv')



if test_df.shape[0] == sub_df.shape[0]:

    sub_df.to_csv('submission.csv',index=False)

else:

    exit(1)