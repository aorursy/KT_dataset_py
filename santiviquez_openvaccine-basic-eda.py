import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

submission = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
print(f"Train Shape: {train.shape}")

print(f"Test Shape: {test.shape}")

print(f"Submission Shape: {submission.shape}")
print(f"Train Columns: {train.columns.values}")

print(f"Test Columns: {test.columns.values}")

print(f"Submission Columns: {submission.columns.values}")
train.head()
train.id.duplicated().sum()
test.id.duplicated().sum()
train.head(1).sequence.values[0]
len(train.head(1).sequence.values[0])
seq_train_lengths = []

for i in range(train.shape[0]):

    seq_train_lengths.append(len(train.sequence.values[i]))





seq_test_lengths = []

for i in range(test.shape[0]):

    seq_test_lengths.append(len(test.sequence.values[i]))
set(seq_train_lengths)
set(seq_test_lengths)
train.seq_length.value_counts()
test.seq_length.value_counts()
Counter(train.head(1).sequence.values[0])
train.structure.head(1).values[0]
Counter(train.structure.head(1).values[0])
train.structure.value_counts()
test.structure.value_counts()
struc_train_lengths = []

for i in range(train.shape[0]):

    struc_train_lengths.append(len(train.structure.values[i]))





struc_test_lengths = []

for i in range(test.shape[0]):

    struc_test_lengths.append(len(test.structure.values[i]))
set(struc_train_lengths)
set(struc_test_lengths)
train.predicted_loop_type.head(1).values[0]
len(train.predicted_loop_type.head(1).values[0])
Counter(train.predicted_loop_type.head(1).values[0])
train.seq_scored.value_counts()
test.seq_scored.value_counts()