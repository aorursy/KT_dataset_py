import numpy as np

import pandas as pd 

import seaborn as sns

from sklearn import preprocessing 

import warnings 

warnings.filterwarnings('ignore')

from collections import Counter

from matplotlib import pyplot as plt



import itertools

#import pymorphy2

import nltk

import re

import string



from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV





from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer





import os
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head(5)
train.info()
train[train["target"] == 0]["text"].values[1]
train[train["target"] == 1]["text"].values[1]
Real_len = train[train['target'] == 1].shape[0]

Not_len = train[train['target'] == 0].shape[0]
bars = pd.DataFrame([Real_len, Not_len], columns=['Size'])

sns.barplot(data=bars, x=bars.index, y = bars.Size)
# word_count

train['word_count'] = train['text'].apply(lambda x: len(str(x).split()))

test['word_count'] = test['text'].apply(lambda x: len(str(x).split()))



# unique_word_count

train['unique_word_count'] = train['text'].apply(lambda x: len(set(str(x).split())))

test['unique_word_count'] = test['text'].apply(lambda x: len(set(str(x).split())))



# url_count

train['url_count'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

test['url_count'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))



# mean_word_length

train['mean_word_length'] = train['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test['mean_word_length'] = test['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



# char_count

train['char_count'] = train['text'].apply(lambda x: len(str(x)))

test['char_count'] = test['text'].apply(lambda x: len(str(x)))



# punctuation_count

train['punctuation_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

test['punctuation_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))



# hashtag_count

train['hashtag_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

test['hashtag_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c == '#']))



# mention_count

train['mention_count'] = train['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

test['mention_count'] = test['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
sns.set_style("whitegrid")

sns.set_context("talk")





meta = ['word_count', 'unique_word_count', 'url_count', 'mean_word_length',

                'char_count', 'punctuation_count', 'hashtag_count', 'mention_count']

disaster = train['target'] == 1



fig, axes = plt.subplots(ncols=2, nrows=len(meta), figsize=(20, 50))



for i, feature in enumerate(meta):

    sns.distplot(train.loc[~disaster][feature], label='Not Disaster', ax=axes[i][0], color='green')

    sns.distplot(train.loc[disaster][feature], label='Disaster', ax=axes[i][0], color='red')



    sns.distplot(train[feature], label='Training', ax=axes[i][1], color='blue')

    sns.distplot(test[feature], label='Test', ax=axes[i][1], color='orange')

    

    for j in range(2):

        axes[i][j].set_xlabel('')

        axes[i][j].tick_params(axis='x', labelsize=12)

        axes[i][j].tick_params(axis='y', labelsize=12)

        axes[i][j].legend()

    

    axes[i][0].set_title(f'{feature} Target Distribution in Training Set', fontsize=13)

    axes[i][1].set_title(f'{feature} Training & Test Set Distribution', fontsize=13)



plt.show()
def prd(num):

    print(train.text[num])

    

prd(12)

prd(100)

prd(25)

prd(20)
l = []

for text in train.text:

    l += text.split()

uniq_words = Counter(l)

print(uniq_words.most_common(50))
# REMAKING WORDS

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



def clean_text(a_text):

    a_text = a_text.lower()

    a_text = re.sub('https?://\S+|www\.\S+', ' ', a_text)

    a_text = re.sub('- full re\x89\S+', ' ', a_text)

    a_text = re.sub('\x89',' ',a_text)

    a_text = re.sub('\.\.\.', ' DOT ', a_text)

    a_text = re.sub('[_(%)-=,>~<;|\\\+\Û\ª\[\]\û\+]', ' ', a_text)

    a_text = re.sub('!+',' ! ', a_text)

    a_text = re.sub('\?+',' ? ', a_text)

    #a_text = re.sub('','',a_text)

    a_text = re.sub('the ',' ',a_text)

    a_text = re.sub(' the ',' ',a_text)

    a_text = re.sub(' a ',' ',a_text)

    a_text = re.sub('a ',' ',a_text)

    a_text = re.sub(' be ',' ',a_text)

    a_text = re.sub(' are ',' ',a_text)

    a_text = re.sub(' was ',' ',a_text)

    a_text = re.sub(' an ',' ',a_text)

    a_text = re.sub('an ',' ',a_text)

    a_text = re.sub(' s ',' ',a_text)

    a_text = re.sub(' via ',' ',a_text)

    a_text = re.sub(':', ' : ', a_text)

    a_text = re.sub('\'','', a_text)

    a_text = re.sub('[\n\t\r]', ' ', a_text)

    a_text = re.sub('<.*?>', ' ', a_text)

    a_text = re.sub('\w*\d\w*', 'NUM', a_text)

    a_text = re.sub('#', ' ', a_text)

    a_text = re.sub('@\S*', ' ', a_text)

    #a_text = re.sub('[%s]' % re.escape(string.punctuation), '', a_text)

    a_text = re.sub(' +', ' ', a_text)

    

    return a_text
train['text'] = train['text'].apply(lambda x: remove_emoji(x))

test['text'] = test['text'].apply(lambda x: remove_emoji(x))

train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))

prd(12)

prd(100)

prd(25)

prd(20)

for i in range(1670,1680):

    prd(i)
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))

test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))

train['text'].head()
def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in nltk.corpus.stopwords.words('english')]

    return words





train['text'] = train['text'].apply(lambda x : remove_stopwords(x))

test['text'] = test['text'].apply(lambda x : remove_stopwords(x))

train.head()
def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



train['text'] = train['text'].apply(lambda x : combine_text(x))

test['text'] = test['text'].apply(lambda x : combine_text(x))

train['text']

train.head()
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test["text"])



## Keeping only non-zero elements to preserve space 

print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])
clf_tfidf = LogisticRegression(C=3.0)

scores = model_selection.cross_val_score(clf_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")

scores
clf_tfidf.fit(train_tfidf, train["target"])
def submission(submission_file_path,model,test_vectors):

    sample_submission = pd.read_csv(submission_file_path)

    sample_submission["target"] = model.predict(test_vectors)

    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

test_vectors=test_tfidf

submission(submission_file_path,clf_tfidf,test_vectors)