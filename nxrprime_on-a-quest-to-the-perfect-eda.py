import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
HOME = '../input/google-quest-challenge/'

train = pd.read_csv(HOME+'train.csv')

test = pd.read_csv(HOME+'test.csv')

sub = pd.read_csv(HOME+'sample_submission.csv')
train.head()
test.head()
distilled_train = train.drop(['qa_id', 'question_title', 'question_body', 'question_user_page', 'question_user_name'], axis=1)

distilled_train = distilled_train.drop(['answer', 'category', 'url', 'answer_user_page', 'answer_user_name', 'host'], axis=1)
distilled_train
distilled_train.corr()
sns.heatmap(distilled_train.corr())
sns.heatmap(train.corr())
sns.clustermap(train.corr(), cmap="mako", robust=True, linewidths=.5)
sns.clustermap(distilled_train.corr(), cmap="viridis", robust=True, linewidths=.5)
sns.clustermap(sub.corr(), cmap="YlGnBu", robust=True, linewidths=.5)
sns.heatmap(sub.corr(), cmap="YlGnBu", robust=True, linewidths=.5)
sns.pairplot(train.head(20))