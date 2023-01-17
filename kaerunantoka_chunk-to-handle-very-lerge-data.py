import gc

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



DATA_DIR = "../input/ashrae-energy-prediction/"

SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")



target_col = "meter_reading"

demo = True

sample = 5
# ===============================

# test_preds is your predictions

# ===============================



test_preds = np.random.randn(41697600).reshape(-1, 1)

print(test_preds.shape)

test_preds[:sample]
if demo:

    chunks = 4000000

    j = 0

    for i, sub in enumerate(pd.read_csv(SUB_PATH, chunksize=chunks, iterator=True)):

        print(i, ":", sub.shape)

        sub[target_col] = test_preds[j:j+chunks]

        j += chunks

        

        if i == 0:

            sub.to_csv('submission.csv', header=True, mode='a', index=False)

        else: 

            sub.to_csv('submission.csv', header=False, mode='a', index=False)

            

        del sub

        gc.collect()        
sub = pd.read_csv('submission.csv')

sub.head(10)