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
pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 100)

from tqdm import tqdm, tqdm_gui

tqdm.pandas(ncols=75) 

from tqdm.notebook import tqdm

from sklearn.utils import shuffle

from bs4 import BeautifulSoup

import re

import warnings

warnings.filterwarnings('ignore')



from nltk.corpus import stopwords

", ".join(stopwords.words('english'))

", ".join(stopwords.words('russian'))

STOPWORDS = set(stopwords.words('russian')) ### need to change while language change



import string



from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

import nltk

from nltk.stem.porter import PorterStemmer

from collections import Counter

cnt = Counter()

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

!pip install git+https://github.com/ssut/py-googletrans.git

from googletrans import Translator

from dask import bag, diagnostics
train = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/train.csv')

test = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/test.csv')
## Number of words in the text

train['Premise_Length'] = train['premise'].apply(lambda x: len(str(x)))

train['Hypothesis_Length'] = train['hypothesis'].apply(lambda x:len(str(x)))

## Number of words in the text

train["premise_num_words"] = train["premise"].apply(lambda x: len(str(x).split()))

train["hypothesis_num_words"] = train["hypothesis"].apply(lambda x: len(str(x).split()))

## Average length of the words in the text ##

train["premise_mean_word_len"] = train["premise"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

train["hypothesis_mean_word_len"] = train["hypothesis"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

## Number of punctuations in the text ##

train["premise_num_punctuations"] =train["premise"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

train["hypothesis_num_punctuations"] =train["hypothesis"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of words in the text

test['Premise_Length'] = test['premise'].apply(lambda x: len(str(x)))

test['Hypothesis_Length'] = test['hypothesis'].apply(lambda x:len(str(x)))

## Number of words in the text

test["premise_num_words"] = test["premise"].apply(lambda x: len(str(x).split()))

test["hypothesis_num_words"] = test["hypothesis"].apply(lambda x: len(str(x).split()))

## Average length of the words in the text ##

test["premise_mean_word_len"] = test["premise"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test["hypothesis_mean_word_len"] = test["hypothesis"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

## Number of punctuations in the text ##

test["premise_num_punctuations"] =test["premise"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["hypothesis_num_punctuations"] =test["hypothesis"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

train


# def Translation(x):

#     translator = Translator()

#     return translator.translate(x).text



# test.premise[test.lang_abv!= 'en']=test.premise[test.lang_abv!= 'en'].apply(lambda x: Translation(x))

# print("here")

# test.hypothesis[test.lang_abv!= 'en']=test.hypothesis[test.lang_abv!= 'en'].apply(lambda x: Translation(x))



# train.premise[train.lang_abv!= 'en']=train.premise[train.lang_abv!= 'en'].apply(lambda x: Translation(x))

# print("here")

# train.hypothesis[train.lang_abv!= 'en']=train.hypothesis[train.lang_abv!= 'en'].apply(lambda x: Translation(x))



def translate(text, dest):

    translator = Translator()

    return translator.translate(text, dest='en').text

def trans_parallel(df, dest):

    premise_bag = bag.from_sequence(df.premise.tolist()).map(translate, dest)

    hypo_bag =  bag.from_sequence(df.hypothesis.tolist()).map(translate, dest)

    with diagnostics.ProgressBar():

        premises = premise_bag.compute()

        hypos = hypo_bag.compute()

    df[['premise', 'hypothesis']] = list(zip(premises, hypos))

    return df

eng = train.loc[train.lang_abv == "en"].copy()

non_eng =  train.loc[train.lang_abv != "en"].copy().pipe(trans_parallel, dest='en')

train = eng.append(non_eng)
eng = test.loc[test.lang_abv == "en"].copy()

non_eng =  test.loc[test.lang_abv != "en"].copy().pipe(trans_parallel, dest='en')

test = eng.append(non_eng)
# train = train.loc[train.lang_abv == "ru"].copy()

# # train = train[['premise', 'hypothesis', 'label']]

# test = test.loc[test.lang_abv == "ru"].copy()

# # test = test[['premise', 'hypothesis', 'label']]

# train.head(10)
def remove_space(text):

    return " ".join(text.split())



def remove_punctuation(text):

    return re.sub("[!#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`]", ' ', text)

#     return re.sub("[!?@#$+%*:()'-.=]", ' ', text)



def remove_dates(text):

    return re.sub("[1234567890]", ' ', text)



def remove_html(text):

    return BeautifulSoup(text, "lxml").text



def remove_url(text):

    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub(r'', text)





def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])



wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatize_words(text):

    pos_tagged_text = nltk.pos_tag(text.split())

    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])



FREQWORDS = set([w for (w, wc) in cnt.most_common(30)])  ## can be changed

def remove_freqwords(text):

    return " ".join([word for word in str(text).split() if word not in FREQWORDS])



n_rare_words = 10

RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

def remove_rarewords(text):

    """custom function to remove the rare words"""

    return " ".join([word for word in str(text).split() if word not in RAREWORDS])



# stemmer = PorterStemmer()

# def stem_words(text):

#     return " ".join([stemmer.stem(word) for word in text.split()])





def clean_text(text):

    text = remove_space(text)

    text = remove_html(text)

    text = remove_url(text)

    text = remove_punctuation(text)

    text = remove_dates(text)

    text = remove_stopwords(text)

    text = lemmatize_words(text)

    text = text.lower()

    text = remove_freqwords(text)

    text = remove_rarewords(text)

#     text = stem_words(text)

    tokens = [w for w in text.split()]

    long_words=[]

    for i in tokens:

        if len(i)>=3:                  #removing short words

            long_words.append(i)   

    return (" ".join(long_words)).strip()





train['premise'] = train.premise.progress_apply(lambda text : clean_text(text))

train['hypothesis'] = train.hypothesis.apply(lambda text : clean_text(text))

test['premise'] = test.premise.progress_apply(lambda text : clean_text(text))

test['hypothesis'] = test.hypothesis.apply(lambda text : clean_text(text))

train.head(100)
test_text = test[['premise', 'hypothesis']].values.tolist()

test_text
train_text = train[['premise', 'hypothesis']].values.tolist()

train_text
from transformers import AutoTokenizer, TFAutoModel

MODEL_NAME = 'jplu/tf-xlm-roberta-large'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)



EPOCHS = 10

MAX_LEN = 80

RATE = 1e-5



## splitting sentences into array of numbers

train_encoded = tokenizer.batch_encode_plus(

    train_text,

    pad_to_max_length=True,

    max_length=MAX_LEN

)
test_encoded = tokenizer.batch_encode_plus(

    test_text,

    pad_to_max_length=True,

    max_length=MAX_LEN

)
# splitting dataset into test and train datasets

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train_encoded['input_ids'], train.label.values, test_size=0.2, random_state=2020)

x_test = test_encoded['input_ids']


params = {

        'objective':'multi:softprob',

        'n_estimators': 1000,

        'num_class': 3,

        'booster':'gbtree',

        'max_depth':9,

        'eval_metric':'mlogloss',

        'learning_rate':0.01, 

        'min_child_weight':1,

        'subsample':0.9,

        'colsample_bytree':0.5,

        'seed':45,

        'reg_lambda':1,

        'reg_alpha':0.01,

        'gamma':0.45,

        'nthread':-1,

}

import xgboost as xgb

d_train = xgb.DMatrix(np.array(x_train), label=y_train)

d_valid = xgb.DMatrix(np.array(x_valid), label=y_valid)

d_test = xgb.DMatrix(np.array(x_test))





watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, maximize=False, verbose_eval=10)
submission = pd.read_csv('/kaggle/input/contradictory-my-dear-watson/sample_submission.csv')

test_preds = clf.predict(d_test)

submission['prediction'] = test_preds.argmax(axis=1)

submission.to_csv('submission.csv', index=False)

submission.head()