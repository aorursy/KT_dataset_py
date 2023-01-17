# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import gc

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
df_train = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')



print('Training Set Shape = {}'.format(df_train.shape))

print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))

print('Test Set Shape = {}'.format(df_test.shape))

print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))
df_train.head(10)
x=df_train.target.value_counts()

sns.barplot(x.index, x)

plt.gca().set_ylabel('tweets')
def count_emoji(text):

    emoji_pattern = re.compile("[:=;<][^a-z ]+")

    #this is a pretty crappy regex but it'll do for now

    return len(emoji_pattern.findall(text))



count_emoji("Omg another Earthquake :( :( :()")
df_train['emoji_only'] = df_train['text'].apply(lambda x: count_emoji(x))
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))

emoji_ct = df_train[df_train['target'] == 1]['emoji_only']

ax1.hist(emoji_ct, color='red')

ax1.set_title('disaster tweets')

emoji_ct = df_train[df_train['target']==0]['emoji_only']

ax2.hist(emoji_ct, color='blue')

ax2.set_title('not disaster tweets')

fig.suptitle('Emojis in tweets')

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=df_train[df_train['target']==1]['text'].str.len()

ax1.hist(tweet_len,color='red')

ax1.set_title('disaster tweets')

tweet_len=df_train[df_train['target']==0]['text'].str.len()

ax2.hist(tweet_len,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()
def count_punctuation(text):

    punct_pattern = re.compile("[!,.?]+")

    #this is a pretty crappy regex but it'll do for now

    punctuation_list = ''.join([item for sublist in punct_pattern.findall(text) for item in sublist])

    return len(punctuation_list)





count_punctuation("Omg another Earthquake!!!!!!")
df_train['punct_ct'] = df_train['text'].apply(lambda x: count_punctuation(x))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_punct=df_train[df_train['target']==1]['punct_ct']

ax1.hist(tweet_punct,color='red')

ax1.set_title('disaster tweets')

tweet_punct=df_train[df_train['target']==0]['punct_ct']

ax2.hist(tweet_punct,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Punctuation in tweets')

plt.show()
!pip install pyspellchecker



from spellchecker import SpellChecker

spell = SpellChecker()

def percentage_spelled_correct(text):

    misspelled_words = spell.unknown(text.split())

    correct_words = len(text.split()) - len(misspelled_words)

    return correct_words / len(text.split())



text = 'corect me pls'

percentage_spelled_correct(text)
df_train['percent_spelled_correctly'] = df_train['text'].apply(lambda x: percentage_spelled_correct(x))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_punct=df_train[df_train['target']==1]['percent_spelled_correctly']

ax1.hist(tweet_punct,color='red')

ax1.set_title('disaster tweets')

tweet_punct=df_train[df_train['target']==0]['percent_spelled_correctly']

ax2.hist(tweet_punct,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Misspellings in tweets')

plt.show()
def count_hashtags(text):

    punct_pattern = re.compile("#[a-zA-Z1-9]+")

    #this is a pretty crappy regex but it'll do for now

    punctuation_list = punct_pattern.findall(text)

    return len(punctuation_list)





count_hashtags("Omg another Earthquake!!!!!! #stuff #things")
df_train['hashtag_count'] = df_train['text'].apply(lambda x: count_hashtags(x))
column='hashtag_count'

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_punct=df_train[df_train['target']==1][column]

ax1.hist(tweet_punct,color='red')

ax1.set_title('disaster tweets')

tweet_punct=df_train[df_train['target']==0][column]

ax2.hist(tweet_punct,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Hashtags in tweets')

plt.show()
def percentage_capitalized(text):

    cap_ct = 0

    uncap_ct = 0

    for char in text:

        if char.isupper():

            cap_ct +=1

        elif char.islower():

            uncap_ct +=1

    return cap_ct / (cap_ct + uncap_ct)





text = 'THIS IS SUPER URGENT!! ok maybe not THAT urgent'

percentage_capitalized(text)
df_train['percent_capitalized'] = df_train['text'].apply(lambda x: count_hashtags(x))
column='percent_capitalized'

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_punct=df_train[df_train['target']==1][column]

ax1.hist(tweet_punct,color='red')

ax1.set_title('disaster tweets')

tweet_punct=df_train[df_train['target']==0][column]

ax2.hist(tweet_punct,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('capitalization in tweets')

plt.show()
def remove_punct_and_lower(text):

    table=str.maketrans('','',string.punctuation)

    return text.lower().translate(table)



spell = SpellChecker()

def correct_spellings(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)



df_train['text'] = df_train['text'].apply(lambda x: remove_punct_and_lower(x))

df_test['text'] = df_test['text'].apply(lambda x: remove_punct_and_lower(x))

print('got here')

# df_train['text'] = df_train['text'].apply(lambda x: correct_spellings(x))

# df_test['text'] = df_test['text'].apply(lambda x: correct_spellings(x))

count_vectorizer = feature_extraction.text.CountVectorizer()



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(df_train["text"][0:5])
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(df_train["text"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(df_test["text"])



## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

clf = linear_model.RidgeClassifier()



scores = model_selection.cross_val_score(clf, train_vectors, df_train["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors, df_train["target"])

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)