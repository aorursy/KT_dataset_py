!pip install emoji --quiet
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.sparse import hstack
from scipy import sparse
from numpy import asarray
from numpy import savetxt


import emoji
import os
import re
import itertools
import math
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import tensorflow_hub as hub



# LOAD DATASETS
df_train = pd.read_csv('/kaggle/input/Quora Question Pairs - Train.csv', index_col='id')
df_test = pd.read_csv('/kaggle/input/Quora Question Pairs - Test.csv',  index_col='id')
df_train.shape
# A list of CONTRACTION_EN from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
CONTRACTION_EN = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "thx": "thanks",
    "lool": "lol"
}

# Thanks to https://www.kaggle.com/rftexas/text-only-kfold-bert
ABBREVATION_EN = {
    "$": " dollar ",
    "€": " euro ",
    "4ao": "for adults only",
    "a.m": "before midday",
    "a3": "anytime anywhere anyplace",
    "aamof": "as a matter of fact",
    "acct": "account",
    "adih": "another day in hell",
    "afaic": "as far as i am concerned",
    "afaict": "as far as i can tell",
    "afaik": "as far as i know",
    "afair": "as far as i remember",
    "afk": "away from keyboard",
    "app": "application",
    "approx": "approximately",
    "apps": "applications",
    "asap": "as soon as possible",
    "asl": "age, sex, location",
    "atk": "at the keyboard",
    "ave.": "avenue",
    "aymm": "are you my mother",
    "ayor": "at your own risk",
    "b&b": "bed and breakfast",
    "b+b": "bed and breakfast",
    "b.c": "before christ",
    "b2b": "business to business",
    "b2c": "business to customer",
    "b4": "before",
    "b4n": "bye for now",
    "b@u": "back at you",
    "bae": "before anyone else",
    "bak": "back at keyboard",
    "bbbg": "bye bye be good",
    "bbc": "british broadcasting corporation",
    "bbias": "be back in a second",
    "bbl": "be back later",
    "bbs": "be back soon",
    "be4": "before",
    "bfn": "bye for now",
    "blvd": "boulevard",
    "bout": "about",
    "brb": "be right back",
    "bros": "brothers",
    "brt": "be right there",
    "bsaaw": "big smile and a wink",
    "btw": "by the way",
    "bwl": "bursting with laughter",
    "c/o": "care of",
    "cet": "central european time",
    "cf": "compare",
    "cia": "central intelligence agency",
    "csl": "can not stop laughing",
    "cu": "see you",
    "cul8r": "see you later",
    "cv": "curriculum vitae",
    "cwot": "complete waste of time",
    "cya": "see you",
    "cyt": "see you tomorrow",
    "dae": "does anyone else",
    "dbmib": "do not bother me i am busy",
    "diy": "do it yourself",
    "dm": "direct message",
    "dwh": "during work hours",
    "e123": "easy as one two three",
    "eet": "eastern european time",
    "eg": "example",
    "embm": "early morning business meeting",
    "encl": "enclosed",
    "encl.": "enclosed",
    "etc": "and so on",
    "faq": "frequently asked questions",
    "fawc": "for anyone who cares",
    "fb": "facebook",
    "fc": "fingers crossed",
    "fig": "figure",
    "fimh": "forever in my heart",
    "ft.": "feet",
    "ft": "featuring",
    "ftl": "for the loss",
    "ftw": "for the win",
    "fwiw": "for what it is worth",
    "fyi": "for your information",
    "g9": "genius",
    "gahoy": "get a hold of yourself",
    "gal": "get a life",
    "gcse": "general certificate of secondary education",
    "gfn": "gone for now",
    "gg": "good game",
    "gl": "good luck",
    "glhf": "good luck have fun",
    "gmt": "greenwich mean time",
    "gmta": "great minds think alike",
    "gn": "good night",
    "g.o.a.t": "greatest of all time",
    "goat": "greatest of all time",
    "goi": "get over it",
    "gps": "global positioning system",
    "gr8": "great",
    "gratz": "congratulations",
    "gyal": "girl",
    "h&c": "hot and cold",
    "hp": "horsepower",
    "hr": "hour",
    "hrh": "his royal highness",
    "ht": "height",
    "ibrb": "i will be right back",
    "ic": "i see",
    "icq": "i seek you",
    "icymi": "in case you missed it",
    "idc": "i do not care",
    "idgadf": "i do not give a damn fuck",
    "idgaf": "i do not give a fuck",
    "idk": "i do not know",
    "ie": "that is",
    "i.e": "that is",
    "ifyp": "i feel your pain",
    "IG": "instagram",
    "iirc": "if i remember correctly",
    "ilu": "i love you",
    "ily": "i love you",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "imu": "i miss you",
    "iow": "in other words",
    "irl": "in real life",
    "j4f": "just for fun",
    "jic": "just in case",
    "jk": "just kidding",
    "jsyk": "just so you know",
    "l8r": "later",
    "lb": "pound",
    "lbs": "pounds",
    "ldr": "long distance relationship",
    "lmao": "laugh my ass off",
    "lmfao": "laugh my fucking ass off",
    "lol": "laughing out loud",
    "ltd": "limited",
    "ltns": "long time no see",
    "m8": "mate",
    "mf": "motherfucker",
    "mfs": "motherfuckers",
    "mfw": "my face when",
    "mofo": "motherfucker",
    "mph": "miles per hour",
    "mr": "mister",
    "mrw": "my reaction when",
    "ms": "miss",
    "mte": "my thoughts exactly",
    "nagi": "not a good idea",
    "nbc": "national broadcasting company",
    "nbd": "not big deal",
    "nfs": "not for sale",
    "ngl": "not going to lie",
    "nhs": "national health service",
    "nrn": "no reply necessary",
    "nsfl": "not safe for life",
    "nsfw": "not safe for work",
    "nth": "nice to have",
    "nvr": "never",
    "nyc": "new york city",
    "oc": "original content",
    "og": "original",
    "ohp": "overhead projector",
    "oic": "oh i see",
    "omdb": "over my dead body",
    "omg": "oh my god",
    "omw": "on my way",
    "p.a": "per annum",
    "p.m": "after midday",
    "pm": "prime minister",
    "poc": "people of color",
    "pov": "point of view",
    "pp": "pages",
    "ppl": "people",
    "prw": "parents are watching",
    "ps": "postscript",
    "pt": "point",
    "ptb": "please text back",
    "pto": "please turn over",
    "qpsa": "what happens",
    "ratchet": "rude",
    "rbtl": "read between the lines",
    "rlrt": "real life retweet",
    "rofl": "rolling on the floor laughing",
    "roflol": "rolling on the floor laughing out loud",
    "rotflmao": "rolling on the floor laughing my ass off",
    "rt": "retweet",
    "ruok": "are you ok",
    "sfw": "safe for work",
    "sk8": "skate",
    "smh": "shake my head",
    "sq": "square",
    "srsly": "seriously",
    "ssdd": "same stuff different day",
    "tbh": "to be honest",
    "tbs": "tablespooful",
    "tbsp": "tablespooful",
    "tfw": "that feeling when",
    "thks": "thank you",
    "tho": "though",
    "thx": "thank you",
    "tia": "thanks in advance",
    "til": "today i learned",
    "tl;dr": "too long i did not read",
    "tldr": "too long i did not read",
    "tmb": "tweet me back",
    "tntl": "trying not to laugh",
    "ttyl": "talk to you later",
    "u": "you",
    "u2": "you too",
    "u4e": "yours for ever",
    "utc": "coordinated universal time",
    "w/": "with",
    "w/o": "without",
    "w8": "wait",
    "wassup": "what is up",
    "wb": "welcome back",
    "wtf": "what the fuck",
    "wtg": "way to go",
    "wtpa": "where the party at",
    "wuf": "where are you from",
    "wuzup": "what is up",
    "wywh": "wish you were here",
    "yd": "yard",
    "ygtr": "you got that right",
    "ynk": "you never know",
    "zzz": "sleeping bored and tired"
}


def remove_contractions(text):
    return CONTRACTION_EN[text.lower()] if text.lower() in CONTRACTION_EN.keys() else text

def remove_abbrevation(text):
    return ABBREVATION_EN[text.lower()] if text.lower() in ABBREVATION_EN.keys() else text

def clean_dataset(text):
    # To lowercase
    text = text.lower()
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    text= ''.join(c for c in text if c <= '\uFFFF') 
    text = text.strip()
    # Remove misspelling words
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    # Remove punctuation
    text = ' '.join(re.sub("[\.\,\!\?\:\;\-\=\/\|\'\(\']", " ", text).split())
    # Remove emoji
    text = emoji.demojize(text)
    text = text.replace(":"," ")
    text = ' '.join(text.split()) 
    text = re.sub("([^\x00-\x7F])+"," ",text)
    # Remove Mojibake (also extra spaces)
    text = ' '.join(re.sub("[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a]", " ", text).split())
    return text

def clean_text(text):
    text = re.sub(r'https?://\S+', '', text) # Remove link
    text = re.sub(r'#', '', text) # Remove hashtag
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'\n',' ', text) # Remove line breaks
    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces
    return text

def find_hashtags(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'

def find_mentions(tweet):
    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'

def find_links(tweet):
    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'

def process_text(df):
    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))
    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))
    df['mentions'] = df['text'].apply(lambda x: find_mentions(x))
    df['links'] = df['text'].apply(lambda x: find_links(x))
    return df
df_train = df_train.fillna('None')
df_test = df_test.fillna('None')
df_train['question1']
df_train['question1'] = df_train['question1'].apply(lambda x: clean_text(x))
df_train['question1'] = df_train['question1'].apply(lambda x : remove_contractions(x))
df_train['question1'] = df_train['question1'].apply(lambda x : clean_dataset(x))
df_train['question1'] = df_train['question1'].apply(lambda x : remove_abbrevation(x))
df_train['question2'] = df_train['question2'].apply(lambda x: clean_text(x))
df_train['question2'] = df_train['question2'].apply(lambda x : remove_contractions(x))
df_train['question2'] = df_train['question2'].apply(lambda x : clean_dataset(x))
df_train['question2'] = df_train['question2'].apply(lambda x : remove_abbrevation(x))
df_train['question1']
'''df_train['is_equal'] = (df_train['question1'] == df_train['question2'])
df_train['len_question1'] = df_train['question1'].apply(lambda x: len(x))
df_train['len_question2'] = df_train['question2'].apply(lambda x: len(x))
df_train['len_diff_question1'] = df_train['len_question1'] - df_train['len_question2'] '''
df_train.head()
y = df_train['is_duplicate']
## Same thing for testing dataset
df_test['question1'] = df_test['question1'].apply(lambda x: clean_text(x))
df_test['question1'] = df_test['question1'].apply(lambda x : remove_contractions(x))
df_test['question1'] = df_test['question1'].apply(lambda x : clean_dataset(x))
df_test['question1'] = df_test['question1'].apply(lambda x : remove_abbrevation(x))
df_test['question2'] = df_test['question2'].apply(lambda x: clean_text(x))
df_test['question2'] = df_test['question2'].apply(lambda x : remove_contractions(x))
df_test['question2'] = df_test['question2'].apply(lambda x : clean_dataset(x))
df_test['question2'] = df_test['question2'].apply(lambda x : remove_abbrevation(x))
'''df_test['is_equal'] = (df_test['question1'] == df_test['question2'])
df_test['len_question1'] = df_test['question1'].apply(lambda x: len(x))
df_test['len_question2'] = df_test['question2'].apply(lambda x: len(x))
df_test['len_diff_question1'] = df_test['len_question1'] - df_test['len_question2']'''
def train_model(train_features,train_target,n_iterations):
    
    model = RandomizedSearchCV(estimator = XGBClassifier(), param_distributions =
                      {
                          'learning_rate': [0.1,0.2,0.3],
                          'n_estimators': [100,150,300],
                          'max_depth': [2,3,4,5,6]
                      },
                      n_iter = n_iterations,
                      scoring = 'accuracy',
                      n_jobs = 5
                      )
    
    model.fit(train_features,train_target)
    
    return model
train_text_q1 = df_train.question1
train_text_q2 = df_train.question2
test_text_q1 = df_test.question1
test_text_q2 = df_test.question2
all_text = pd.concat([train_text_q1, train_text_q2,test_text_q1, test_text_q2 ])
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    norm='l2',
    min_df=0,
    smooth_idf=False,
    max_features=20000)
word_vectorizer.fit(all_text)
train_word_features1 = word_vectorizer.transform(train_text_q1)
train_word_features2 = word_vectorizer.transform(train_text_q2)
test_word_features1 = word_vectorizer.transform(test_text_q1)
test_word_features2 = word_vectorizer.transform(test_text_q2)

train_features = hstack([train_word_features1, train_word_features2]).tocsr()
test_features = hstack([test_word_features1, test_word_features2]).tocsr()
model_tdif = train_model(train_features, y,test_features,10)
y_pred = model_tdif.predict(test_features) 
y_pred= y_pred.round().astype(np.int)
# save to csv file
savetxt('submission_quora1.csv', y_pred, delimiter=',')
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
X_train_embeddings1 = embed(df_train.question1)
X_train_embeddings2 = embed(df_train.question2)
X_train_embeddings = np.concatenate((X_train_embeddings1.numpy(), X_train_embeddings2.numpy()), axis=1)
del X_train_embeddings1
del X_train_embeddings2
gc.collect()
X_feat=sparse.csr_matrix(X_train_embeddings)
del X_train_embeddings
gc.collect()
y.shape, X_feat.shape
model_google_hub = XGBClassifier()
model_google_hub.fit(X_feat,y)
model_google_hub
gc.collect()
X_test_embeddings1 = embed(df_test.question1)
X_test_embeddings2 = embed(df_test.question2)
X_test_embeddings = np.concatenate((X_test_embeddings1.numpy(), X_test_embeddings2.numpy()), axis=1)
del X_test_embeddings1
del X_test_embeddings2
gc.collect()
X_test_embeddings.shape
X_test = sparse.csr_matrix(X_test_embeddings)
del X_test_embeddings
gc.collect()
y_pred_2 = model_google_hub.predict(X_test) 
y_pred2 = y_pred_2.round().astype(np.int)
y_pred2
# save to csv file
savetxt('submission_quora2.csv', y_pred2, delimiter=',')