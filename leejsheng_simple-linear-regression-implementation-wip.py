import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
# Explore the files

train_file = "../input/train.csv"

test_file = "../input/test.csv"

data_description_file = "../input/data_description.txt"

sample_submission_file = "../input/sample_submission.csv"
# load files as data frames

train_pd = pd.read_csv(train_file)

test_pd = pd.read_csv(test_file)

sample_submission_pd = pd.read_csv(sample_submission_file)
# explore the description of the data

with open(data_description_file,'r') as data_description:

    print(data_description.read())
train_pd.describe()

train_pd.info()
train_pd.head()
test_pd.head()
sample_submission_pd.head()
features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'PoolArea']