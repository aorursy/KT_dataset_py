!ls -lh ../input/data-science-bowl-2019/
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import altair as alt

from altair.vega import v5

from IPython.display import HTML



%matplotlib inline

plt.rc('figure', figsize=(15.0, 8.0))
root = '../input/data-science-bowl-2019/'

train = pd.read_csv(root + 'train.csv')

train_labels = pd.read_csv(root + 'train_labels.csv')

specs = pd.read_csv(root + 'specs.csv')

test = pd.read_csv(root + 'test.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')
train.head()
train.info()
train['installation_id'].unique().shape # total 17000 installations in train data
specs.head()
specs.info()
train_labels.head()
train_labels.info()
test.head()
test['installation_id'].unique().shape # total 1000 installations for which we have to predict
sample_submission.head()
sample_submission.shape # to predict for 1000 installations
train = train.merge(specs, on='event_id')

train_labels = train.merge(train_labels, on=['game_session', 'installation_id']) # returns only type == Assessments
train.shape, train_labels.shape
train.head()
train_labels.head()
fig, ax = plt.subplots(figsize=(10, 10))

plot = sns.countplot(y="type", data=train, palette=['navy', 'darkblue', 'blue', 'dodgerblue']).set_title('train type count', fontsize=16)

plt.yticks(fontsize=14)

plt.xlabel("Count", fontsize=15)

plt.ylabel("type", fontsize=15)

plt.show(plot)
fig, ax = plt.subplots(figsize=(10, 10))

plot = sns.countplot(y="type", data=test, palette=['navy', 'darkblue', 'blue', 'dodgerblue']).set_title('test type count', fontsize=16)

plt.yticks(fontsize=14)

plt.xlabel("Count", fontsize=15)

plt.ylabel("type", fontsize=15)

plt.show(plot)
train_by_type = train.groupby('type')

train_clip = train_by_type.get_group('Clip')

train_game = train_by_type.get_group('Game')

train_activity = train_by_type.get_group('Activity')

train_assessment = train_by_type.get_group('Assessment')
train_clip.head()
train_game.head()
train_activity.head()
train_assessment.head()
sample_submission.to_csv('submission.csv', index=False)