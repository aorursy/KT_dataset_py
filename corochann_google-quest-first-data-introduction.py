import gc

import os

from pathlib import Path

import random

import sys



from tqdm import tqdm_notebook as tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

datadir = Path('/kaggle/input/google-quest-challenge')



# Read in the data CSV files

train = pd.read_csv(datadir/'train.csv')

test = pd.read_csv(datadir/'test.csv')

sample_submission = pd.read_csv(datadir/'sample_submission.csv')
print('train', train.shape)

print('test', test.shape)

print('sample_submission', sample_submission.shape)
sample_submission.head()
sample_submission.columns
feature_columns = [col for col in train.columns if col not in sample_submission.columns]

print('Feature columns: ', feature_columns)
train[feature_columns].head()
train0 = train.iloc[0]



print('URL           : ', train0['url'])

print('question_title: ', train0['question_title'])

print('question_body : ', train0['question_body'])
print('answer        : ', train0['answer'])
train[['url', 'question_user_name', 'question_user_page', 'answer_user_name', 'answer_user_page', 'url', 'category', 'host']]
target_cols = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
train[target_cols]
fig, axes = plt.subplots(6, 5, figsize=(18, 15))

axes = axes.ravel()

bins = np.linspace(0, 1, 20)



for i, col in enumerate(target_cols):

    ax = axes[i]

    sns.distplot(train[col], label=col, kde=False, bins=bins, ax=ax)

    # ax.set_title(col)

    ax.set_xlim([0, 1])

    ax.set_ylim([0, 6079])

plt.tight_layout()

plt.show()

plt.close()
train.isna().sum()
test.isna().sum()
train_category = train['category'].value_counts()

test_category = test['category'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

train_category.plot(kind='bar', ax=axes[0])

axes[0].set_title('Train')

test_category.plot(kind='bar', ax=axes[1])

axes[1].set_title('Test')

print('Train/Test category distribution')
from wordcloud import WordCloud





def plot_wordcloud(text, ax, title=None):

    wordcloud = WordCloud(max_font_size=None, background_color='white',

                          width=1200, height=1000).generate(text_cat)

    ax.imshow(wordcloud)

    if title is not None:

        ax.set_title(title)

    ax.axis("off")
print('Training data Word Cloud')



fig, axes = plt.subplots(1, 3, figsize=(16, 18))



text_cat = ' '.join(train['question_title'].values)

plot_wordcloud(text_cat, axes[0], 'Question title')



text_cat = ' '.join(train['question_body'].values)

plot_wordcloud(text_cat, axes[1], 'Question body')



text_cat = ' '.join(train['answer'].values)

plot_wordcloud(text_cat, axes[2], 'Answer')



plt.tight_layout()

fig.show()
print('Test data Word Cloud')



fig, axes = plt.subplots(1, 3, figsize=(16, 18))



text_cat = ' '.join(test['question_title'].values)

plot_wordcloud(text_cat, axes[0], 'Question title')



text_cat = ' '.join(test['question_body'].values)

plot_wordcloud(text_cat, axes[1], 'Question body')



text_cat = ' '.join(test['answer'].values)

plot_wordcloud(text_cat, axes[2], 'Answer')



plt.tight_layout()

fig.show()
fig, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(train[target_cols].corr(), ax=ax)
train_question_user = train['question_user_name'].unique()

test_question_user = test['question_user_name'].unique()



print('Number of unique question user in train: ', len(train_question_user))

print('Number of unique question user in test : ', len(test_question_user))

print('Number of unique question user in both train & test : ', len(set(train_question_user) & set(test_question_user)))
train_answer_user = train['answer_user_name'].unique()

test_answer_user = test['answer_user_name'].unique()



print('Number of unique answer user in train: ', len(train_answer_user))

print('Number of unique answer user in test : ', len(test_answer_user))

print('Number of unique answer user in both train & test : ', len(set(train_answer_user) & set(test_answer_user)))
print('Number of unique user in both question & anser in train  : ', len(set(train_answer_user) & set(train_question_user)))

print('Number of unique user in both question & anser in train  : ', len(set(test_answer_user) & set(test_question_user)))
def char_count(s):

    return len(s)



def word_count(s):

    return s.count(' ')
train['question_title_n_chars'] = train['question_title'].apply(char_count)

train['question_title_n_words'] = train['question_title'].apply(word_count)

train['question_body_n_chars'] = train['question_body'].apply(char_count)

train['question_body_n_words'] = train['question_body'].apply(word_count)

train['answer_n_chars'] = train['answer'].apply(char_count)

train['answer_n_words'] = train['answer'].apply(word_count)



test['question_title_n_chars'] = test['question_title'].apply(char_count)

test['question_title_n_words'] = test['question_title'].apply(word_count)

test['question_body_n_chars'] = test['question_body'].apply(char_count)

test['question_body_n_words'] = test['question_body'].apply(word_count)

test['answer_n_chars'] = test['answer'].apply(char_count)

test['answer_n_words'] = test['answer'].apply(word_count)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.distplot(train['question_title_n_chars'], label='train', ax=axes[0])

sns.distplot(test['question_title_n_chars'], label='test', ax=axes[0])

axes[0].legend()

sns.distplot(train['question_title_n_words'], label='train', ax=axes[1])

sns.distplot(test['question_title_n_words'], label='test', ax=axes[1])

axes[1].legend()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.distplot(train['question_body_n_chars'], label='train', ax=axes[0])

sns.distplot(test['question_body_n_chars'], label='test', ax=axes[0])

axes[0].legend()

sns.distplot(train['question_body_n_words'], label='train', ax=axes[1])

sns.distplot(test['question_body_n_words'], label='test', ax=axes[1])

axes[1].legend()
train['question_body_n_chars'].clip(0, 5000, inplace=True)

test['question_body_n_chars'].clip(0, 5000, inplace=True)

train['question_body_n_words'].clip(0, 1000, inplace=True)

test['question_body_n_words'].clip(0, 1000, inplace=True)



fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.distplot(train['question_body_n_chars'], label='train', ax=axes[0])

sns.distplot(test['question_body_n_chars'], label='test', ax=axes[0])

axes[0].legend()

sns.distplot(train['question_body_n_words'], label='train', ax=axes[1])

sns.distplot(test['question_body_n_words'], label='test', ax=axes[1])

axes[1].legend()
train['answer_n_chars'].clip(0, 5000, inplace=True)

test['answer_n_chars'].clip(0, 5000, inplace=True)

train['answer_n_words'].clip(0, 1000, inplace=True)

test['answer_n_words'].clip(0, 1000, inplace=True)



fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.distplot(train['answer_n_chars'], label='train', ax=axes[0])

sns.distplot(test['answer_n_chars'], label='test', ax=axes[0])

axes[0].legend()

sns.distplot(train['answer_n_words'], label='train', ax=axes[1])

sns.distplot(test['answer_n_words'], label='test', ax=axes[1])

axes[1].legend()
from scipy.spatial.distance import cdist



def calc_corr(df, x_cols, y_cols):

    arr1 = df[x_cols].T.values

    arr2 = df[y_cols].T.values

    corr_df = pd.DataFrame(1 - cdist(arr2, arr1, metric='correlation'), index=y_cols, columns=x_cols)

    return corr_df
number_feature_cols = ['question_title_n_chars', 'question_title_n_words', 'question_body_n_chars', 'question_body_n_words', 'answer_n_chars', 'answer_n_words']

# train[number_feature_cols].corrwith(train[target_cols], axis=0)



corr_df = calc_corr(train, target_cols, number_feature_cols)
corr_df
fig, ax = plt.subplots(figsize=(25, 5))

sns.heatmap(corr_df, ax=ax)
num_question = train['question_user_name'].value_counts()

num_answer = train['answer_user_name'].value_counts()



train['num_answer_user'] = train['answer_user_name'].map(num_answer)

train['num_question_user'] = train['question_user_name'].map(num_question)

test['num_answer_user'] = test['answer_user_name'].map(num_answer)

test['num_question_user'] = test['question_user_name'].map(num_question)



# # map is done by train data, we need to fill value for user which does not appear in train data...

# test['num_answer_user'].fillna(1, inplace=True)

# test['num_question_user'].fillna(1, inplace=True)
number_feature_cols = ['num_answer_user', 'num_question_user']

# train[number_feature_cols].corrwith(train[target_cols], axis=0)



corr_df = calc_corr(train, target_cols, number_feature_cols)
fig, ax = plt.subplots(figsize=(30, 2))

sns.heatmap(corr_df, ax=ax)