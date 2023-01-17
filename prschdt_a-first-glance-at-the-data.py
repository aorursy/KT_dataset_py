import os



print(os.listdir('../input/nlp-getting-started'))
import pandas as pd

import numpy as np



# plotting

import matplotlib.pyplot as plt

import seaborn as sns



import re, string
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')

sample = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
train.head()
train = train.drop(columns='id')
test_ids = test.id

test = test.drop(columns='id')
sns.countplot(train.target).set_title('Target variable distribution')
(100.0 * train.isna().sum() / train.shape[0]).to_frame(name='percentage').sort_values(by='percentage')
def is_keyword_in(data):

    if data.keyword in data.text.split():

        return 1

    else:

        return 0
train['keyword_appears'] = train[['keyword', 'text']].dropna().apply(is_keyword_in, axis=1)
print('Percentage of keyword appearence in disasters')

100.0 * train[train.target == 1].keyword_appears.value_counts(normalize=True).to_frame(name='percentage')
train[train.target == 1].keyword_appears.value_counts(normalize=True).plot(kind='bar').set_title('Does keyword appear in real disasters?')
print('Percentage of keyword appearence in non-disasters')

100.0 * train[train.target == 0].keyword_appears.value_counts(normalize=True).to_frame(name='percentage')
train[train.target == 0].keyword_appears.value_counts(normalize=True).plot(kind='bar').set_title('Does keyword appear in non disasters?')
pd.crosstab(train.target, train.keyword_appears)
train.location.dropna().value_counts().to_frame(name='count')
def get_num_words(data):

    return len(data.split())
# Number of characters

train['num_chars'] = train.text.apply(len)



# Number of words

train['num_words'] = train.text.apply(get_num_words)
train.num_chars.describe()
sns.boxplot(x='target', y='num_chars', data=train[['num_chars', 'target']]).set_title('Number of characters')
train.num_words.describe()
sns.boxplot(x='target', y='num_words', data=train[['num_words', 'target']]).set_title('Number of words')
mentions = 0



for tweet in train.text.values:

    words = tweet.split()

    for w in words:

        if w[0] == '@':

            mentions += 1



print('Number of mentions:', mentions)

print('Number of tweets:', train.shape[0])
def has_mention(data):

    mentions = 0

    for word in data.text.split():

        if word[0] == '@':

            mentions += 1

    

    return mentions
train['mention'] = train.apply(has_mention, axis=1)
print('Percentage of mentions in disasters')

100.0 * train[train.target == 1].mention.value_counts(normalize=True).to_frame(name='percentage')
train[train.target == 1].mention.value_counts(normalize=True).plot(kind='bar').set_title('Do mentions appear in disasters?')
print('Percentage of mentions in non-disasters')

100.0 * train[train.target == 0].mention.value_counts(normalize=True).to_frame(name='percentage')
train[train.target == 0].mention.value_counts(normalize=True).plot(kind='bar').set_title('Do mentions appear in non-disasters?')
def remove_URL(data):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',data)



def remove_html(data):

    html = re.compile(r'<.*?>')

    return html.sub(r'',data)



# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b

def remove_emoji(data):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    

    return emoji_pattern.sub(r'', data)



def remove_punct(data):

    table=str.maketrans('','',string.punctuation)

    return data.translate(table)



def make_lower(data):

    return data.lower()



def remove_mentions(data):

    words = data.split()

    

    words = [word for word in words if word[0] != '@']

    return ' '.join(words)



def clean_data(data, drop=False, test=False, lowercase=False, correct=False, rmv_mentions=False):

    data.text = data.text.apply(remove_URL)

    data.text = data.text.apply(remove_html)

    data.text = data.text.apply(remove_emoji)

    data.text = data.text.apply(remove_punct)

    

    if lowercase:

        data.text = data.text.apply(make_lower)

    

    if correct:

        data.text = data.text.apply(correct_spellings)

    

    if rmv_mentions:

        data.text = data.text.apply(remove_mentions)

    

    if drop and test:

        return data[['text']]

    elif drop:

        return data[['text', 'target']]

    

    return data
%%time

train = clean_data(train, drop=True, lowercase=True, rmv_mentions=True)
train.head()