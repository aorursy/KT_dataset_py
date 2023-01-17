%reload_ext autoreload

%autoreload 2

%matplotlib inline
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# load data

train_df = pd.read_csv('../input/train_relationships.csv')

test_df = pd.read_csv('../input/sample_submission.csv')
train_df.head()
test_df.head()