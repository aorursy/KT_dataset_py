import numpy as np 

import pandas as pd 

# These are the main data files which contain the gameplay events

train =pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

# This file demonstrates how to compute the ground truth for the assessments in the training set.

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
train.head()
train_labels.head()
train.sample(5)
# This file gives the specification of the various event types.

specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
# sample_submission.csv A sample submission in the correct format.

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')