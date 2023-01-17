# Importing all needed libraries 



!python -m spacy download en_core_web_md



import re

import numpy as np

import pandas as pd

import spacy

import en_core_web_md

from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import train_test_split

import sklearn.metrics as metrics



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Here are the functions I use for preprocessing tweets and extracting features from texts



def prepros(tweet):

    """

    Tokenize input text, removes stop words and punctuations.

    

    Returns:

    -List of tokens

    """

    tweet = re.sub(r'[^0-9a-zA-Z ]', '', tweet).lower()

    doc = nlp(tweet)

    words = [token.lemma_ for token in doc if not (token.is_stop) and not (token.is_punct)]

    return words



def words_counter(text):

    n_words = len([word for word in text if not len(word) == 0])

    return n_words



def tag_counter(tweet):

    return len([x for x in tweet if '@' in x])



def average_len(tweet):

    lengths = [len(x) for x in tweet]

    return sum(lengths)/len(tweet)



def hashtag_counter(tweet):

    return len([x for x in tweet if '#' in x])



def ents_counter(doc):

    return len(doc.ents)



def punct_counter(doc):

    return len([token for token in doc if token.is_punct])



def link_detector(tweet):

    lst = [x for x in tweet if 'http' in x]

    if len(lst) == 0:

        return np.nan

    else:

        return lst[0]

    

def verb_counter(tweet):

    lst = [x for x in tweet if x.pos_ == 'VERB']

    return len(lst)



def noun_counter(tweet):

    lst = [x for x in tweet if x.pos_ == 'NOUN']

    return len(lst)
test_data = pd.read_csv("../input/nlp-getting-started/test.csv")

train_data = pd.read_csv("../input/nlp-getting-started/train.csv")



# Load language data for Spacy text processing

nlp = en_core_web_md.load()
# Transforming text to tokenized text and Doc type for further manipulations

train_data['tok_text'] = train_data.text.apply(lambda x: x.split(' '))

train_data['doc'] = train_data.text.apply(lambda x: nlp(x))



# applying functions to train data

train_data['n_words'] = train_data.tok_text.apply(words_counter)

train_data['n_tags'] = train_data.tok_text.apply(tag_counter)

train_data['n_hashtags'] = train_data.tok_text.apply(hashtag_counter)

train_data['avg_len_word'] = train_data.tok_text.apply(average_len)

train_data['n_ents'] = train_data.doc.apply(ents_counter)

train_data['links'] = train_data.tok_text.apply(link_detector)

train_data['verbs'] = train_data.doc.apply(verb_counter)

train_data['nouns'] = train_data.doc.apply(noun_counter)



train_data.fillna(-999, inplace=True)  # CatBoost handles outliers, so we don't need to worry about NaN values



features_to_drop = ['id', 'text', 'tok_text', 'doc']



X = train_data.drop(features_to_drop + ['target'], axis=1)

y = train_data.target



# The same manipulations with test data

test_data['tok_text'] = train_data.text.apply(lambda x: x.split(' '))

test_data['doc'] = train_data.text.apply(lambda x: nlp(x))



test_data['n_words'] = test_data.tok_text.apply(words_counter)

test_data['n_tags'] = test_data.tok_text.apply(tag_counter)

test_data['n_hashtags'] = test_data.tok_text.apply(hashtag_counter)

test_data['avg_len_word'] = test_data.tok_text.apply(average_len)

test_data['n_ents'] = test_data.doc.apply(ents_counter)

test_data['links'] = test_data.tok_text.apply(link_detector)

test_data['verbs'] = test_data.doc.apply(verb_counter)

test_data['nouns'] = test_data.doc.apply(noun_counter)



test_data.fillna(-999, inplace=True)



X_test = test_data.drop(features_to_drop, axis=1)



display(train_data)

display(test_data)
# Splits training data into training set and validation set

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)



# Creates Pool class for convinient usage in CatBoost Classifier

train_Pool = Pool(X_train, y_train, cat_features=[0, 1, 7])
# Training our Classifier

model = CatBoostClassifier(verbose=False, 

                           custom_metric='F1')

                           

model.fit(train_Pool)
y_pred_eval = model.predict(X_eval)



F1_eval = metrics.f1_score(y_eval, y_pred_eval)

print(F1_eval)
feat_importance = model.get_feature_importance(train_Pool)



plt.figure(figsize=(13, 10))

sns.set_style('darkgrid')

sns.barplot(x=feat_importance, y=X_train.columns).set_title('Feature Importance')
model_key = CatBoostClassifier(verbose=False,

                             custom_metric='F1')



X_train_key = pd.DataFrame(X_train.keyword) # I have 'KeyError: 0' using Series, so let it be a DF



model_key.fit(Pool(X_train_key, y_train, cat_features=[0]))



print(metrics.f1_score(y_eval, model_key.predict(X_eval.keyword)))