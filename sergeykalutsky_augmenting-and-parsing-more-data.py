import numpy as np

import pandas as pd 

import os

import datetime as dt
!ls ../input/jigsaw-toxic-comment-classification-challenge
data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

data.head()
data['toxic'] = data[data.columns[2:]].sum(axis=1)

toxic = data[data['toxic'] > 0]

not_toxic = data[data['toxic'] == 0]

print(f'toxic comments: {toxic.shape[0]}, normal: {not_toxic.shape[0]}')
!ls ../input/wikipedia-talk-labels-personal-attacks/
data_commets = pd.read_csv('../input/wikipedia-talk-labels-personal-attacks/attack_annotated_comments.csv')

data_attack =  pd.read_csv('../input/wikipedia-talk-labels-personal-attacks/attack_annotations.csv')
data_attack.drop(columns=data_attack.columns[1:-1], inplace=True)

summery = data_attack.groupby(['rev_id']).sum()

data = data_commets.set_index('rev_id').join(summery)
data.head()
toxic = data[data['attack'] > 0]

not_toxic = data[data['attack'] == 0]

print(f'toxic comments: {toxic.shape[0]}, normal: {not_toxic.shape[0]}')
!ls ../input/wikipedia-talk-corpus-sample/
# The whole corpus is too large to load in kernel, here is a small sample

data = pd.read_csv('../input/wikipedia-talk-corpus-sample/chunk_0.tsv', sep='\t')

data.head()
!pip install psaw
# Example of general use

from psaw import PushshiftAPI

api = PushshiftAPI()



# The `search_comments` and `search_submissions` methods return generator objects

gen = api.search_submissions(limit=100)

results = list(gen)



# There are 2 main attributes we may be interested in:

# title - provides the title of a submission

print(results[1].title)

# selftext - provides main text, if text exists

print(results[1].selftext)
# Start time

start_epoch=int(dt.datetime(2017, 1, 1).timestamp())

# Found submissions

shit_reddit_says = list(api.search_submissions(after=start_epoch,

                                               subreddit='ShitRedditSays',

                                               filter=['url','author', 'title', 'subreddit'],

                                               limit=10))



# Some questionable comments

for i in range (6):

    print(shit_reddit_says[i].title, '\n')
# Start time

start_epoch=int(dt.datetime(2018, 1, 1).timestamp())

# Found submissions

lgbt = list(api.search_submissions(after=start_epoch,

                                               subreddit='lgbt',

                                               filter=['url','author', 'title', 'subreddit'],

                                               limit=10))



# Some positive comments

for i in range (6):

    print(lgbt[i].title, '\n')