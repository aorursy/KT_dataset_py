import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from collections import defaultdict

import re

from nltk.corpus import stopwords

import string

from emoji import UNICODE_EMOJI
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
train_1_df = train_df[train_df['target'] == 1]

train_0_df = train_df[train_df['target'] == 0]
to_plot = 100*train_df['target'].value_counts()/train_df.shape[0]



# Set the width and height of the figure

plt.figure(figsize=(10,6))



# Add title

plt.title("target distribution")



sns.barplot(x=to_plot.index, y=to_plot.values)



# Add label for vertical axis

plt.ylabel("% of data")

train_0_df['text'].values[:10]
train_1_df['text'].values[:10]
def create_word_frequency(df):

    word_frequency = defaultdict(int)

    for text in df['text']:

        for word in text.split():

            word = re.sub(r'[^a-z ]', '', word.lower())

            if word:

                word_frequency[word] += 1

    word_frequency = {k: v for k, v in sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)}

    return pd.DataFrame(word_frequency.items(), columns=['words', 'counts'])
fig, axs = plt.subplots(2, 2, figsize=(20,12))

fig.suptitle('Word Frequency Before Removing Stop-Words', fontsize=16)



axs[0, 0].set_title('Train Word Frequency')

sns.barplot(x='words', y='counts', data=create_word_frequency(train_df).head(10), ax=axs[0,0])



axs[0, 1].set_title('Test Word Frequency')

sns.barplot(x='words', y='counts', data=create_word_frequency(test_df).head(10), ax=axs[0,1])



axs[1, 0].set_title('Emergency Tweets Word Frequency')

sns.barplot(x='words', y='counts', data=create_word_frequency(train_1_df).head(10), ax=axs[1,0])



axs[1, 1].set_title('Non-Emergency Tweets Word Frequency')

sns.barplot(x='words', y='counts', data=create_word_frequency(train_0_df).head(10), ax=axs[1,1])
stop = set(stopwords.words('english')) 

def create_non_stop_word_frequency(df):

    word_frequency = defaultdict(int)

    for text in df['text']:

        for word in text.split():

            word = re.sub(r'[^a-z ]', '', word.lower())

            if word and word not in stop:

                word_frequency[word] += 1

    word_frequency = {k: v for k, v in sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)}

    return pd.DataFrame(word_frequency.items(), columns=['words', 'counts'])
fig, axs = plt.subplots(2, 2, figsize=(20,12))

fig.suptitle('Word Frequency After Removing Stop-Words', fontsize=16)



axs[0, 0].set_title('Train Word Frequency')

sns.barplot(x='words', y='counts', data=create_non_stop_word_frequency(train_df).head(10), ax=axs[0,0])



axs[0, 1].set_title('Test Word Frequency')

sns.barplot(x='words', y='counts', data=create_non_stop_word_frequency(test_df).head(10), ax=axs[0,1])



axs[1, 0].set_title('Emergency Tweets Word Frequency')

sns.barplot(x='words', y='counts', data=create_non_stop_word_frequency(train_1_df).head(10), ax=axs[1,0])



axs[1, 1].set_title('Non-Emergency Tweets Word Frequency')

sns.barplot(x='words', y='counts', data=create_non_stop_word_frequency(train_0_df).head(10), ax=axs[1,1])
punct = set(string.punctuation)

def create_punctuation_frequency(df):

    punct_frequency = defaultdict(int)

    for text in df['text']:

        for char in text:

            if char in punct:

                punct_frequency[char] += 1

    punct_frequency = {k: v for k, v in sorted(punct_frequency.items(), key=lambda item: item[1], reverse=True)}

    return pd.DataFrame(punct_frequency.items(), columns=['punctuation', 'counts'])
fig, axs = plt.subplots(2, 2, figsize=(20,12))

fig.suptitle('Punctuation Frequency Before Cleaning', fontsize=16)



axs[0, 0].set_title('Train Punctuation Frequency')

sns.barplot(x='punctuation', y='counts', data=create_punctuation_frequency(train_df).head(10), ax=axs[0,0])



axs[0, 1].set_title('Test Punctuation Frequency')

sns.barplot(x='punctuation', y='counts', data=create_punctuation_frequency(test_df).head(10), ax=axs[0,1])



axs[1, 0].set_title('Emergency Tweets Punctuation Frequency')

sns.barplot(x='punctuation', y='counts', data=create_punctuation_frequency(train_1_df).head(10), ax=axs[1,0])



axs[1, 1].set_title('Non-Emergency Tweets Punctuation Frequency')

sns.barplot(x='punctuation', y='counts', data=create_punctuation_frequency(train_0_df).head(10), ax=axs[1,1])
punct = set(string.punctuation)

def create_cleaned_punctuation_frequency(df):

    punct_frequency = defaultdict(int)

    for text in df['text']:

        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        text = re.sub(r'<.*?>', '', text)

        emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

        text = emoji_pattern.sub(r'', text)

        for char in text:

            if char in punct:

                punct_frequency[char] += 1

    punct_frequency = {k: v for k, v in sorted(punct_frequency.items(), key=lambda item: item[1], reverse=True)}

    return pd.DataFrame(punct_frequency.items(), columns=['punctuation', 'counts'])
fig, axs = plt.subplots(2, 2, figsize=(20,12))

fig.suptitle('Punctuation Frequency After Cleaning', fontsize=16)



axs[0, 0].set_title('Train Punctuation Frequency')

sns.barplot(x='punctuation', y='counts', data=create_cleaned_punctuation_frequency(train_df).head(10), ax=axs[0,0])



axs[0, 1].set_title('Test Punctuation Frequency')

sns.barplot(x='punctuation', y='counts', data=create_cleaned_punctuation_frequency(test_df).head(10), ax=axs[0,1])



axs[1, 0].set_title('Emergency Tweets Punctuation Frequency')

sns.barplot(x='punctuation', y='counts', data=create_cleaned_punctuation_frequency(train_1_df).head(10), ax=axs[1,0])



axs[1, 1].set_title('Non-Emergency Tweets Punctuation Frequency')

sns.barplot(x='punctuation', y='counts', data=create_cleaned_punctuation_frequency(train_0_df).head(10), ax=axs[1,1])
def create_hashtag_frequency(df):

    hashtags_frequency = defaultdict(int)

    for text in df['text']:

        text = re.findall(r'#[0-9A-Za-z]+', text)

        for hashtag in text:

            if hashtag:

                hashtags_frequency[hashtag.strip('#')] += 1

    hashtags_frequency = {k: v for k, v in sorted(hashtags_frequency.items(), key=lambda item: item[1], reverse=True)}

    return pd.DataFrame(hashtags_frequency.items(), columns=['hashtags', 'counts'])
fig, axs = plt.subplots(2, 2, figsize=(20,12))

fig.suptitle('Hashtag Frequency After Cleaning', fontsize=16)



axs[0, 0].set_title('Train Hashtag Frequency')

sns.barplot(x='hashtags', y='counts', data=create_hashtag_frequency(train_df).head(10), ax=axs[0,0])



axs[0, 1].set_title('Test Hashtag Frequency')

sns.barplot(x='hashtags', y='counts', data=create_hashtag_frequency(test_df).head(10), ax=axs[0,1])



axs[1, 0].set_title('Emergency Tweets Hashtag Frequency')

sns.barplot(x='hashtags', y='counts', data=create_hashtag_frequency(train_1_df).head(10), ax=axs[1,0])



axs[1, 1].set_title('Non-Emergency Tweets Hashtag Frequency')

sns.barplot(x='hashtags', y='counts', data=create_hashtag_frequency(train_0_df).head(10), ax=axs[1,1])
def create_mention_frequency(df):

    mentions_frequency = defaultdict(int)

    for text in df['text']:

        text = re.findall(r'@[_0-9A-Za-z]+', text)

        for mention in text:

            if mention:

                mentions_frequency[mention.strip('@')] += 1

    mentions_frequency = {k: v for k, v in sorted(mentions_frequency.items(), key=lambda item: item[1], reverse=True)}

    return pd.DataFrame(mentions_frequency.items(), columns=['mentions', 'counts'])
fig, axs = plt.subplots(2, 2, figsize=(20,12))

fig.suptitle('Mention Frequency After Cleaning', fontsize=16)



axs[0, 0].set_title('Train Mention Frequency')

sns.barplot(x='mentions', y='counts', data=create_mention_frequency(train_df).head(10), ax=axs[0,0])



axs[0, 1].set_title('Test Mention Frequency')

sns.barplot(x='mentions', y='counts', data=create_mention_frequency(test_df).head(10), ax=axs[0,1])



axs[1, 0].set_title('Emergency Tweets Mention Frequency')

sns.barplot(x='mentions', y='counts', data=create_mention_frequency(train_1_df).head(10), ax=axs[1,0])



axs[1, 1].set_title('Non-Emergency Tweets Mention Frequency')

sns.barplot(x='mentions', y='counts', data=create_mention_frequency(train_0_df).head(10), ax=axs[1,1])
punct = set(string.punctuation)

stop = set(stopwords.words('english'))

def clean(text):

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    emoji_pattern = re.compile("["

                       u"\U0001F600-\U0001F64F"  # emoticons

                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                       u"\U0001F680-\U0001F6FF"  # transport & map symbols

                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                       u"\U00002702-\U000027B0"

                       u"\U000024C2-\U0001F251"

                       "]+", flags=re.UNICODE)

    text = emoji_pattern.sub(r'', text)

    text = re.sub(r'[^a-z ]', '', text.lower())

    return text
def create_average_df(df):

    average_df = df.copy()

    average_df['char_amount'] = df.apply(lambda x: len(x['text']), axis=1)

    average_df['word_amount'] = df.apply(lambda x: len(x['text'].split()), axis=1)

    average_df['average_word_length'] = df.apply(lambda x: np.mean(list(map(lambda x: len(x), clean(x['text']).split()))), axis=1)

    average_df['punct_amount'] = df.apply(lambda x: len([p for p in x['text'] if p in punct]), axis=1)

    average_df['mentions_amount'] = df.apply(lambda x: re.subn(r'#[0-9A-Za-z]+', '', x['text'])[1], axis=1)

    average_df['hashtags_amount'] = df.apply(lambda x: re.subn(r'@[_0-9A-Za-z]+', '', x['text'])[1], axis=1)

    average_df['URL_amount'] = df.apply(lambda x: re.subn(r'https?://\S+|www\.\S+', '', x['text'])[1], axis=1)

    average_df['html_amount'] = df.apply(lambda x: re.subn(r'<.*?>', '', x['text'])[1], axis=1)

    return average_df
train_avg_df = create_average_df(train_df)

test_avg_df = create_average_df(test_df)
dists = ['char_amount', 'word_amount', 'average_word_length', 'punct_amount']

hists = ['mentions_amount', 'hashtags_amount', 'URL_amount', 'html_amount']

fig, axs = plt.subplots(8, 2, figsize=(20,48))

axs[0, 0].set_title('train vs test')

axs[0, 1].set_title('Emergeny vs Not')

for ax, col in zip(axs[0:4, 0], dists):

    sns.distplot(train_avg_df[col], ax=ax, color='red')

    sns.distplot(test_avg_df[col], ax=ax, color='green')

    ax.legend(labels=['train', 'test'])

    ax.set_ylabel(col, rotation=0, size='large')

    ax.set_xlabel('')

    ax.yaxis.set_label_coords(-0.15, 0.5)

    

for ax, col in zip(axs[4:8, 0], hists):

    sns.distplot(train_avg_df[col], ax=ax, color='red', hist=True, kde=False)

    sns.distplot(test_avg_df[col], ax=ax, color='green', hist=True, kde=False)

    ax.legend(labels=['train', 'test'])

    ax.set_ylabel(col, rotation=0, size='large')

    ax.set_xlabel('')

    ax.yaxis.set_label_coords(-0.15, 0.5)

    

for ax, col in zip(axs[0:4, 1], dists):

    sns.distplot(train_avg_df[train_avg_df['target'] == 1][col], ax=ax, color='red')

    sns.distplot(train_avg_df[train_avg_df['target'] == 0][col], ax=ax, color='green')

    ax.set_xlabel('')

    ax.legend(labels=['Emergency', 'Not'])

    

for ax, col in zip(axs[4:8, 1], hists):

    sns.distplot(train_avg_df[train_avg_df['target'] == 1][col], ax=ax, color='red', hist=True, kde=False)

    sns.distplot(train_avg_df[train_avg_df['target'] == 0][col], ax=ax, color='green', hist=True, kde=False)

    ax.set_xlabel('')

    ax.legend(labels=['Emergency', 'Not'])



fig.tight_layout()