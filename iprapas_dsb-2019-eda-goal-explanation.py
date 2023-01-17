import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import gc

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import lightgbm as lgb

from numba import jit

from sklearn.metrics import cohen_kappa_score, confusion_matrix

import time

import seaborn as sns

%%time 

train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')

test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')

train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
print("Train shape",train_df.shape)

print("Test shape",test_df.shape)

print("Train labels shape",train_labels_df.shape)

print("Specs shape",specs_df.shape)
train_df.head()
train_labels_df.head()
specs_df.head()
train_labels_df['title'].unique()
train_labels_df['accuracy_group'].value_counts().sort_index().plot.bar()

plt.title('Accuracy group - target')
plt.figure(figsize=(15,10))

sns.countplot(x='title', hue='accuracy_group',data=train_labels_df)

plt.show()
plt.figure(figsize=(15,10))

sns.countplot(x='title', hue='num_correct',data=train_labels_df)

plt.show()
f, axes = plt.subplots(1, 2,figsize=(20,10))

sns.countplot(x='world',data=train_df, ax=axes[0])

axes[0].set_title="world(train)"

sns.countplot(x='world',data=test_df, ax=axes[1])

axes[1].set_title="world(test)"

plt.show()
f, axes = plt.subplots(1, 2,figsize=(35,10))

sns.countplot(x='event_code',data=train_df, ax=axes[0])

axes[0].set_title="event_code(train)"

sns.countplot(x='event_code',data=test_df, ax=axes[1])

axes[1].set_title="event_code(test)"

plt.show()
train_df.type.unique()
type_order = train_df.type.unique()

f, axes = plt.subplots(1, 2,figsize=(20,10))

sns.countplot(x='type',data=train_df, order=type_order, ax=axes[0])

axes[0].set_title="type(train)"

sns.countplot(x='type',data=test_df, order=type_order, ax=axes[1])

axes[1].set_title="type(test)"

plt.show()
title_order = train_df.title.unique()

f, axes = plt.subplots(1, 2,figsize=(30,10))

sns.countplot(y='title',data=train_df, ax=axes[0], order=title_order)

axes[0].set(title="title(train)")

sns.countplot(y='title',data=test_df, ax=axes[1],order=title_order)

axes[1].set(title="title(test)")

plt.show()