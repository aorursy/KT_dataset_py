!pip install pycontractions

!pip install symspellpy
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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import xgboost as xgb

from tqdm import tqdm

import time

import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import f1_score



from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.util import ngrams

from nltk.stem import PorterStemmer





import pkg_resources

from symspellpy.symspellpy import SymSpell

from symspellpy import SymSpell, Verbosity

#Contraction Import

from pycontractions import Contractions
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")

# bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")



symspell_segmenter = SymSpell(max_dictionary_edit_distance=1, prefix_length=8)

symspell_segmenter.load_dictionary(dictionary_path, term_index=0, count_index=1)



# sym_spell_misspelled = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# sym_spell_misspelled.load_dictionary(dictionary_path, term_index=0, count_index=1)

# sym_spell_misspelled.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
cont = Contractions(api_key="glove-twitter-100")

cont.load_models()
"""Let's load the data files"""

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

train.head()
def to_lower(text):

    text = text.lower()

    return text



#Version Edit: Susanth D.

def remove_url(text):

    url = re.compile(r'https?://\S+|www\.\S+|pic.twitter.com\S+')

    return url.sub(r' ',text)



#Version Edit: Sonam D.

def remove_punct(text):

    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    return text



#Version Edit: Susanth D.

def remove_special_ucchar(text):

    text = re.sub('&.*?;', ' ', text)

    return text



#Version Edit: Sonam D.

def remove_numbers(text):

    text = re.sub(r'\d+', ' ', text)

    return text



#Version Edit: Sonam D.

def remove_mentions(text):

    text = re.sub(r'@\w*', ' ', text)

    return text



#Version Edit: Susanth D.

def handle_unicode(text):

    text = text.encode('ascii', 'replace').decode('utf-8')

    return text



#Version Edit: Sonam D.

def remove_punctuations(text):

    text = re.sub(r'([^A-Za-z \t])|(\w+:\/\/\S+)', ' ', text)

    return text



#Version Edit: Susanth D.

def remove_square_bracket(text):

    text = re.sub('\[.*?\]', ' ', text)

    return text



#Version Edit: Susanth D.

def remove_angular_bracket(text):

    text = re.sub('\<.*?\>+', ' ', text)

    return text



#Version Edit: Susanth D.

def remove_newline(text):

    text = re.sub('\n', ' ', text)

    return text



#Version Edit: Susanth D.

def remove_words_with_numbers(text):

    text = re.sub('\w*\d\w*', ' ', text)

    return text

    

#Version Edit: Susanth D.

def hashtag_to_words(text):

    hashtag_list = re.findall(r"#\w+",text)

    for hashtag in hashtag_list:

        hashtag = re.sub(r'#', '', hashtag)

        text = re.sub(hashtag, symspell_segmenter.word_segmentation(hashtag).segmented_string, text)

    text = re.sub(r'#', ' ', text)

    return text



#Version Edit: Susanth D.

def remove_stopwords(text):

    text_tokens=word_tokenize(text)

    textop = ''

    for token in text_tokens:

        if token not in stopwords.words('english'):

            textop = textop + token + ' '

    return textop



#Version Edit: Sonam D.

def stemming_text(text):

    stemmer= PorterStemmer()

    text_tokens=word_tokenize(text)

    textop = ''

    for token in text_tokens:

        textop = textop + stemmer.stem(token) + ' '

    return textop



#Version Edit: Sonam D.

def lemmatization(text):

    lemmatizer=WordNetLemmatizer()

    text_tokens=word_tokenize(text)

    textop = ''

    for token in text_tokens:

        textop = textop + lemmatizer.lemmatize(token) + ' '

    return textop



#Version Edit: Saurabh M.

def removeRepeated(tweet):

    prev = ''

    tweet_new = ''

    for c in tweet:

        caps = False

        if c.isdigit():

            tweet_new += c

            continue

        if c.isalpha() == True:

            if ord(c) >= 65 and ord(c)<=90:

                caps = True

            c = c.lower()

            if c == prev:

                count += 1

            else:

                count = 1

                prev = c

            if count >= 3:

                continue

            if caps == True:

                tweet_new += c.upper()

            else:

                tweet_new += c

        else:

            tweet_new += c

    return tweet_new



def Expand_Contractions(text):

    return list(cont.expand_texts([text]))[0]
def count_chars(text):

    new_text = text.apply(lambda x : list(x)).explode()

    return new_text.unique().shape[0]



def count_words(text):

    new_text = text.apply(lambda x : x.split(' ')).explode()

    return new_text.unique().shape[0]



def preprocess_pipeline(steps, col, df):

    new_col = df[col]

    char_count_before = 0

    word_count_before = 0

    char_count_after = 0

    word_count_after = 0

    for each_step in steps:

        char_count_before = count_chars(new_col)

        word_count_before = count_words(new_col)

        new_col = new_col.apply(each_step)

        char_count_after = count_chars(new_col)

        word_count_after = count_words(new_col)

        print("Preprocessing step: ",each_step.__name__)

        print("Unique Char Count ---> Before: %d | After: %d"%(char_count_before, char_count_after))

        print("Unique Word Count ---> Before: %d | After: %d"%(word_count_before, word_count_after))

    

    return new_col

        
%%time

train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')





pipeline = []



pipeline.append(handle_unicode)

# pipeline.append(to_lower)

pipeline.append(remove_newline)

pipeline.append(remove_url)

pipeline.append(remove_special_ucchar)

pipeline.append(hashtag_to_words)

pipeline.append(remove_mentions)

# pipeline.append(remove_square_bracket)

# pipeline.append(remove_angular_bracket)

# pipeline.append(Expand_Contractions)

pipeline.append(remove_words_with_numbers)

# pipeline.append(remove_punctuations)

pipeline.append(remove_punct)

# pipeline.append(remove_numbers)

pipeline.append(removeRepeated)

pipeline.append(remove_stopwords)

# pipeline.append(stemming_text)

pipeline.append(lemmatization)

pipeline.append(to_lower)



print("For Training data:")

train['processed_text'] = preprocess_pipeline(pipeline, 'text', train)

print("\nFor Testing data:")

test['processed_text'] = preprocess_pipeline(pipeline, 'text', test)

train.head()
chars = train['processed_text'].apply(lambda x : list(x)).explode()

chars.unique()
print(train['text'].iloc[233])

print(train['processed_text'].iloc[233])
u, idx = np.unique(train['processed_text'], return_index=True)

train = train.iloc[idx]
tweet_len = train['processed_text'].apply(len)

print(tweet_len.max())
xtrain, xvalid, ytrain, yvalid = train_test_split(train.processed_text, 

                                                  train.target,

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)

print(xtrain.shape)

print(xvalid.shape)
full_text = pd.concat([train["processed_text"], test["processed_text"]])
# Appling CountVectorizer()

count_vectorizer = CountVectorizer()

count_vectorizer.fit(full_text)

xtrain_vectors = count_vectorizer.transform(xtrain)

xvalid_vectors = count_vectorizer.transform(xvalid)
# Appling TFIDF

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2), norm='l2')

tfidf.fit(full_text)

xtrain_tfidf = tfidf.transform(xtrain)

xvalid_tfidf = tfidf.transform(xvalid)
# # Fitting a simple xgboost on TFIDF

# clf = xgb.XGBClassifier(max_depth=60, subsample=1, learning_rate=0.07, colsample_bytree=0.8, reg_lambda=0.1, reg_alpha=0.1,\

#                        gamma=1)

# clf.fit(xtrain_tfidf.tocsc(), ytrain)

# predictions = clf.predict(xvalid_tfidf.tocsc())



# print('XGBClassifier on TFIDF')

# print ("f1_score :", np.round(f1_score(yvalid, predictions),5))
%%time

# Fitting a simple xgboost on CountVec

clf = xgb.XGBClassifier(max_depth=200, subsample=1, learning_rate=0.07, reg_lambda=0.1, reg_alpha=0.1,\

                       gamma=1)

clf.fit(xtrain_vectors, ytrain)



predictions = clf.predict(xvalid_vectors)

print('XGBClassifier on CountVectorizer')

print ("f1_score :", np.round(f1_score(yvalid, predictions),5))



predictions = clf.predict(xtrain_vectors)

print('XGBClassifier on CountVectorizer')

print ("Training set f1_score :", np.round(f1_score(ytrain, predictions),5))
# %%time

# # Fitting a simple xgboost on CountVec

# clf = xgb.XGBClassifier(max_depth=200, subsample=0.8, learning_rate=0.07, 

#                         colsample_bytree=0.8, reg_lambda=0.1, reg_alpha=0.1, gamma=1)

# clf.fit(xtrain_vectors, ytrain)



# predictions = clf.predict(xvalid_vectors)

# print('XGBClassifier on CountVectorizer')

# print ("f1_score :", np.round(f1_score(yvalid, predictions),5))



# predictions = clf.predict(xtrain_vectors)

# print('XGBClassifier on CountVectorizer')

# print ("Training set f1_score :", np.round(f1_score(ytrain, predictions),5))
# %%time

# #'''For XGBC, the following hyperparameters are usually tunned.'''

# #'''https://xgboost.readthedocs.io/en/latest/parameter.html'''



# seed = 500

# XGB_model = xgb.XGBClassifier(

#            n_estimators=800,

#            verbose = 1)





# XGB_params = {'max_depth' : (60),

#               'reg_alpha':  (0.1, 0.09, 0.2),

#               'reg_lambda': (0.1, 0),

#               'learning_rate': (0.07, 0.06, 0.08),

# #               'colsample_bytree': (0.5, 1),

#               'gamma': (0, 1),

#              'subsample': [1],

#              'random_state':[seed]}



# grid_search = GridSearchCV(estimator = XGB_model, param_grid = XGB_params, cv = 3, 

#                              verbose = 3,

#                              scoring = 'f1', n_jobs = -1)

# grid_search.fit(xtrain_vectors, ytrain)

# XGB_best_params, XGB_best_score = grid_search.best_params_, grid_search.best_score_

# print('XGB best params:{} & best_score:{:0.5f}' .format(XGB_best_params, XGB_best_score))
# test_vectors = tfidf.transform(test.processed_text)
test_vectors = count_vectorizer.transform(test.processed_text)
# best_model = grid_search.best_estimator_

# test_pred = best_model.predict(test_vectors)

test_pred = clf.predict(test_vectors)

submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)