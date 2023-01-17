# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train.info()
test.info()
len(train)
train.sample(5, random_state=42)
sum(train.target) / len(train)
train_keywords = train.keyword.value_counts()

test_keywords = test.keyword.value_counts()
print(list(train_keywords.index))
print(len(train_keywords))

print(len(test_keywords))
list(train_keywords.index.sort_values()) == list(test_keywords.index.sort_values())
train_keywords[:20]
test_keywords[:20]
train_keywords[-20:]
test_keywords[-20:]
def count_matches(a, b):

  n_match = 0

  for word in a:

    if word in b:

      n_match += 1

  return n_match
top_train_keywords = set(train_keywords[:100].index)

top_test_keywords = list(test_keywords[:100].index)

bottom_train_keywords = set(train_keywords[-100:].index)

bottom_test_keywords = list(test_keywords[-100:].index)
count_matches(top_test_keywords, top_train_keywords)
count_matches(bottom_test_keywords, bottom_train_keywords)
count_matches(top_test_keywords, bottom_train_keywords)
count_matches(bottom_test_keywords, top_train_keywords)
train_locations = train.location.value_counts()

test_locations = test.location.value_counts()
train_locations[:20]
test_locations[:20]
train_locations[-10:]
test_locations[-10:]
count_matches(list(test_locations.index), set(train_locations.index))
count_matches(list(test_locations[:100].index), set(train_locations[:100].index))
train_ids = train.id.value_counts()

test_ids = test.id.value_counts()
train_ids[:5]
test_ids[:5]
train_ids[-5:]
test_ids[-5:]
count_matches(list(test.id), set(train.id))
points = train.sample(2000, replace=False)

plt.scatter(points.id, points.target, alpha=0.05)

plt.xlabel('id')

plt.ylabel('target')

plt.show()
sns.distplot(train[train.target==0].id, label='0')

sns.distplot(train[train.target==1].id, label='1')

plt.xlabel('id')

plt.ylabel('density')

plt.show()
train[train.target==0].id.mean(), train[train.target==0].id.std(), train[train.target==0].id.skew(), train[train.target==0].id.kurt()
train[train.target==1].id.mean(), train[train.target==1].id.std(), train[train.target==1].id.skew(), train[train.target==1].id.kurt()