import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

import random

import os

os.listdir('../input')
train = pd.read_csv('../input/train.csv')
train_full = train
cols = ['target',

    'severe_toxicity',

    'obscene',

    'threat',

    'insult',

    'identity_attack',

    'sexual_explicit',

    'male', 'female','homosexual_gay_or_lesbian','christian','jewish','muslim','black',

    'white','psychiatric_or_mental_illness'

]

print('Numbers of columns of interest: ', len(cols))
for col in cols:

    train_full = train_full[~pd.isnull(train_full[col])]

print('Proportion of dataset with no missing value: ', len(train_full)/len(train))
print('Number of samples: ', len(train_full))

train_full.to_csv('train_full.csv')
train_full.head()
train_full.columns
f, axes = plt.subplots(4, 4,figsize=(20, 15))



color = ['b','g','c','k']



i = 0

j = 0

for name in cols:

    if j == 4:

        j = 0

        i +=1

    sns.distplot(train_full[[name]], kde = False, ax=axes[i][j])

    axes[i][j].set_yscale('log')

    axes[i][j].set_title(name)

    j += 1
sns.pairplot(train_full[['target','sexual_explicit','obscene']])
sns.pairplot(train_full[['target','threat','identity_attack']])
sns.pairplot(train_full[['target','severe_toxicity','insult']])
sns.pairplot(train_full[['target','black','white','psychiatric_or_mental_illness']])
sns.pairplot(train_full[['target','male','female','homosexual_gay_or_lesbian']])
ax = sns.pairplot(train_full[['target','christian','jewish','muslim']])