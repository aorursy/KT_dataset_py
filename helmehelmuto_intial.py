import numpy as np

import pandas as pd

import os
import matplotlib.pyplot as plt

%matplotlib inline



train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

submission_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')



train_df=train_df.set_index('Date')

train_df.shape, test_df.shape, submission_df.shape

train_df