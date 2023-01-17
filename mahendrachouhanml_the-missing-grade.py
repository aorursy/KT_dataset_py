# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import json

import pandas as pd

import ast
# load the training Data

all_traning_records = []

traning_path = "/kaggle/input/training.json"

with open(traning_path) as train_records:

    for train_record in train_records:

        all_traning_records.append(train_record.rstrip())

train_files = all_traning_records[1:]

final_train_file = []

for train_record in train_files:

    final_train_file.append(ast.literal_eval(train_record))

train_df = pd.DataFrame(final_train_file).fillna(0).drop(['serial'], axis=1)

train_df.head()
# load Test json

all_test_records = []

testing_path = "/kaggle/input/sample-test.in.json"

with open(testing_path) as train_records:

    for train_record in train_records:

        all_test_records.append(train_record.rstrip())

test_files = all_test_records[1:]

final_test_file = []

for test_record in test_files:

    final_test_file.append(ast.literal_eval(test_record))

test_df = pd.DataFrame(final_test_file).fillna(0).drop(['serial'], axis=1)

test_df.head()
# acutal results

import numpy as np

all_test_results = []

testing_path = "/kaggle/input/sample-test.out.json"

with open(testing_path) as train_records:

    for train_record in train_records:

        all_test_results.append(np.float64(train_record.rstrip()))

all_test_results[:5]


Y = train_df[["Mathematics"]]

X = train_df.drop(["Mathematics"], axis=1)
# train model for X and Y

from sklearn.linear_model import LinearRegression

reg = LinearRegression(normalize=True).fit(X, Y, )
reg.score(test_df, all_test_results)