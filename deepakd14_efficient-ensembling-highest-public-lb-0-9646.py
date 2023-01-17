import numpy as np

import pandas as pd



import numpy as np

import pandas as pd 

import os 

test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

first = pd.read_csv('../input/output-of-best-public-submission/submission_best.csv')

second = pd.read_csv('../input/output-of-best-public-submission/submission_first.csv')

third = pd.read_csv('../input/output-of-best-public-submission/submission_post_process.csv')

submission = first.copy()

second.head()
arg1 = (2/3)*first['target']

arg2 = (1/6)*second['target']

arg3 = (1/6)*third['target']



submission['target'] = arg1 + arg2 + arg3

submission.to_csv('submission.csv', index=False)

submission.head()