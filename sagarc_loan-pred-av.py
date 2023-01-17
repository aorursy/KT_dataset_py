import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
sample_submission = pd.read_csv("../input/sample_submission.csv")

test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")
#copying the original dataframe into another dataframe

train=train_df.copy()

test=test_df.copy()
#Columns in train data

train_col=train.columns.values

print(f'Columns in train data:\n{train_col}')
test_col=test.columns.values

print(f'Columns in test data are:\n{test_col}')
# Datatypes of Columns

train.dtypes
#Number of unique categories each categorical variable is having

train.select_dtypes('object').apply(pd.Series.nunique,axis=0)
train_cat=set(train.select_dtypes("object").columns.values)

print(train_cat)

train_cat.difference("Loan_ID")