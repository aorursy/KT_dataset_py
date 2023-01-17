import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
sample_submission = pd.read_csv('../input/submission-transactions/submission.csv')

sample_submission.to_csv('submission.csv', index=False)