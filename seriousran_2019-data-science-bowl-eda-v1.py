import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ROOT = '../input/data-science-bowl-2019/'

train_df = pd.read_csv(ROOT + 'train.csv')

train_labels_df = pd.read_csv(ROOT + 'train_labels.csv')

specs_df = pd.read_csv(ROOT + 'specs.csv')

test_df = pd.read_csv(ROOT + 'test.csv')

sample_submission = pd.read_csv(ROOT + 'sample_submission.csv')
print(train_df.shape)

train_df.head()
train_sub_df = train_df.sample(n=1000000, random_state=2019)
train_sub_df['installation_id'].value_counts()
numerical = ['timestamp', 'game_time']

categorical = ['event_id', 'game_session', 'installation_id', 'event_code', 'title', 'type', 'world']

dictionary = ['event_data']
print(train_labels_df.shape)

train_labels_df.head()
train_label_CB_df = train_labels_df[train_labels_df['title']=='Cart Balancer (Assessment)']

train_label_CF_df = train_labels_df[train_labels_df['title']=='Cauldron Filler (Assessment)']

train_label_MS_df = train_labels_df[train_labels_df['title']=='Mushroom Sorter (Assessment)']

train_label_CS_df = train_labels_df[train_labels_df['title']=='Chest Sorter (Assessment)']

train_label_BM_df = train_labels_df[train_labels_df['title']=='Bird Measurer (Assessment)']
for df in [train_label_CB_df, train_label_CF_df, train_label_MS_df, train_label_CS_df, train_label_BM_df]:

    df['accuracy'] = df['accuracy'].apply(lambda x: round(x, 2))
sns.set(rc={'figure.figsize':(20,8)})

sns.countplot(train_label_CB_df['accuracy'])
sns.set(rc={'figure.figsize':(20,8)})

sns.countplot(train_label_CF_df['accuracy'])
sns.set(rc={'figure.figsize':(20,8)})

sns.countplot(train_label_MS_df['accuracy'])
sns.set(rc={'figure.figsize':(20,8)})

sns.countplot(train_label_CS_df['accuracy'])
sns.countplot(train_label_BM_df['accuracy'])