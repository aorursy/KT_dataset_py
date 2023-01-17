

%matplotlib inline

import pandas as pd

import seaborn as sns

import matplotlib as plt

df_act_test = pd.read_csv('../input/act_test.csv')

df_act_train = pd.read_csv('../input/act_train.csv')

df_people = pd.read_csv('../input/people.csv')

df_sample_submission = pd.read_csv('../input/sample_submission.csv')
# TODO

# Create features by person

#   ex: average outcome for person, filling in mean for missing values

# try with svm