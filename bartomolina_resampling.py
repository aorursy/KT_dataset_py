import numpy as np

import pandas as pd



# load the train and test data files

train = pd.read_csv("../input/feature-exploration-and-dataset-preparation/train_clean_standarized.csv", index_col=0)
# let's see how imbalanced is our TARGET feature

print(train.TARGET.value_counts())

ax = train.TARGET.value_counts().plot(kind='bar', title='Customer satisfaction')

ax.set_xticklabels(['satisfied', 'unsatisfied']);
# split the data between satisfied and unsatisfied customers

train_satisfied = train[train.TARGET == 0]

train_unsatisfied = train[train.TARGET == 1]



# undersample the majority class to 20000 instances

train_satisfied_under = train_satisfied.sample(20000)



# combine the two classes

train_resampled = pd.concat([train_satisfied_under, train_unsatisfied]);
# let's see how imbalanced is our TARGET feature in the resampled dataset

print(train_resampled.TARGET.value_counts())

ax = train_resampled.TARGET.value_counts().plot(kind='bar', title='Customer satisfaction')

ax.set_xticklabels(['satisfied', 'unsatisfied']);
train_resampled.to_csv('train_resampled.csv')