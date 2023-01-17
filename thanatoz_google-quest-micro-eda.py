from pathlib import Path

import os

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
RANDOM_SEED = 123

BASE=Path('../input/google-quest-challenge')

for i in os.walk(os.path.join(BASE)):

    print(i)
train_df=pd.read_csv(BASE/'train.csv')

train_df.tail()
test_df=pd.read_csv(BASE/'test.csv')

test_df.tail()
# Viewing the value of first row of the training data

for i,j in train_df.iloc[0].items():

#     print(i.ljust(30),j)

    print('-'*80)

    print(i)

    print('-'*80)

    print(j,'\n')
# Plotting the channels where the Training queries comes from in the data

fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

width = 0.4

train_df.host.value_counts().plot(kind='bar', color='blue', ax=ax, width=width, position=1)

test_df.host.value_counts().plot(kind='bar', color='red', ax=ax, width=width, position=0)

ax.set_xlabel('Sites')

ax.set_ylabel('Question Counts')
# Plotting the channels where the testing queries comes from in the data

plt.figure(figsize=(16,5))

width = 0.4

test_df.host.value_counts().plot(kind='bar', color='red', width=width)
# Plotting the category occurance of the queries

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

width = 0.2

train_df.category.value_counts().plot(kind='bar', color='blue', ax=ax, width=width, position=1, legend=True)

test_df.category.value_counts().plot(kind='bar', color='red', ax=ax, width=width, position=0, legend=True)

ax.set_xlabel('Sites')

ax.set_ylabel('Question Counts')
# Finding the size of longest query in the table

sentence_len = train_df.answer.apply(lambda x: len(x))

sentence_len.max()
# Prints column with longest query

train_df[sentence_len==22636]