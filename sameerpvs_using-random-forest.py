!pip install pyspellchecker



import numpy as np 

import pandas as pd 

from sklearn import feature_extraction, preprocessing

from sklearn.ensemble import RandomForestClassifier

import re

import math

import string

import nltk

from nltk.tokenize import word_tokenize

from spellchecker import SpellChecker





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
def remove_url(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
test_tweet = train_df['text'][100]

print(test_tweet)

test_tweet = remove_url(test_tweet)

test_tweet
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
print(test_tweet + "\U0001F600")

test_tweet = remove_emoji(test_tweet + "\U0001F600")

test_tweet
def remove_punc(text):

    return text.translate(str.maketrans('', '', string.punctuation))        
print(test_tweet)

test_tweet = remove_punc(test_tweet)

test_tweet
def tokenization(text):

    text = re.split('\W+', text)

    return text
print(test_tweet)

test_tweet = tokenization(test_tweet)

print(test_tweet)
stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):

    text = [word for word in text if word not in stopword]

    return text
print(test_tweet)

test_tweet = remove_stopwords(test_tweet)

print(test_tweet)
ps = nltk.PorterStemmer()

def stemming(text):

    text = [ps.stem(word) for word in text]

    return text



spell = SpellChecker()

def spelling(text):

    text = [spell.correction(x) for x in text]

    return(text)
for datas in [train_df,test_df]:

    datas['text'] = datas['text'].apply(lambda x : remove_url(x))

    datas['text'] = datas['text'].apply(lambda x : remove_emoji(x))

    datas['text'] = datas['text'].apply(lambda x : remove_punc(x))

    datas['text'] = datas['text'].apply(lambda x : tokenization(x.lower()))

    datas['text'] = datas['text'].apply(lambda x : remove_stopwords(x))

    datas['text'] = datas['text'].apply(lambda x : ' '.join(x))

print(train_df['text'][100])
count_vec = feature_extraction.text.CountVectorizer()

train_vec = count_vec.fit_transform(train_df['text'])

test_vec = count_vec.transform(test_df['text'])



train_vec[100].todense()
# from sklearn.model_selection import RandomizedSearchCV

# random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)], #num of trees in forest

#                'max_features': ['auto', 'sqrt'],                                                  #num of features at each split

#                'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],                     #max depth of each tree

#                'min_samples_split': [2, 5, 10],                                                   #min num of samples to split a node

#                'min_samples_leaf': [1, 2, 4],                                                     #min num of samples at each leaf node

#                'bootstrap': [True, False]}



# model = RandomForestClassifier()

# print("Beginning tuning")

# model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# model_random.fit(train_vec, train_label)
model = RandomForestClassifier(n_estimators = 100,

                              min_samples_split = 5,

                              min_samples_leaf = 2,

                              max_features = 'auto',

                              max_depth = None,

                              bootstrap =  True)

model.fit(train_vec, train_df['target'])
preds = model.predict(count_vec.transform(test_df['text']))
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = preds.astype(int)

sample_submission
sample_submission.to_csv("submission.csv", index=False)