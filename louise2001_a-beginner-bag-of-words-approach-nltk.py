# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def jaccard(str1, str2): 

    str1, str2 = str(str1), str(str2)

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

sample_submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
train.head()
v1 = train.loc[train.sentiment=='neutral', 'text'].values.tolist()

v2 = train.loc[train.sentiment=='neutral', 'selected_text'].values.tolist()

np.mean([jaccard(w1, w2) for w1, w2 in zip(v1, v2)])
test.head()
sample_submission.head()
isNeutral = test.loc[test['sentiment'] == 'neutral', 'textID'].values.tolist()

def get_selected_text_neutral(textID, df=test):

    if textID in isNeutral:

        return df.loc[df.textID==textID, 'text'].values.tolist()[0]

    else:

        return np.nan

def treat_neutral(sample_submission):

    sample_submission['selected_text'] = sample_submission['textID'].apply(get_selected_text_neutral, df=test)

    return sample_submission

sample_submission = treat_neutral(sample_submission)

sample_submission.head()
from nltk import pos_tag, ngrams

# nltk.download('sentiwordnet')

from nltk.corpus import sentiwordnet as swn, wordnet as wn

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer, PorterStemmer

lemmatizer = WordNetLemmatizer()

ps = PorterStemmer()
import string

filtre = [wn.NOUN, wn.ADJ, wn.ADV, wn.VERB]
def penn_to_wn(tag):

    """

    Convert between the PennTreebank tags to simple Wordnet tags

    """

    if tag.startswith('J'):

        return wn.ADJ

    elif tag.startswith('N'):

        return wn.NOUN

    elif tag.startswith('R'):

        return wn.ADV

    elif tag.startswith('V'):

        return wn.VERB

    else:

        return None

        

def get_sentiment(word, tag, verbose=0):

    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """

    wn_tag = penn_to_wn(tag)

    if wn_tag not in filtre:

        return []



    lemma = lemmatizer.lemmatize(word, pos=wn_tag)

    if verbose:

        print(f'Lemmatizer : {lemma}')

    if not lemma:

        return []



    synsets = wn.synsets(word, pos=wn_tag)

    if verbose:

        print(f'Synsets : {synsets}')

    if not synsets:

        return []



    swn_synset_pos = []

    swn_synset_neg = []

    for synset in synsets:

        swn_synset = swn.senti_synset(synset.name())

        if verbose:

            print(f'Pos score : {swn_synset.pos_score()}, Neg score : {swn_synset.neg_score()}')

        swn_synset_pos.append(swn_synset.pos_score())

        swn_synset_neg.append(swn_synset.neg_score())

    return [np.mean(swn_synset_pos),np.mean(swn_synset_neg)]#,swn_synset.obj_score()



def robustify(text=''):

    if type(text) != str:

        try:

            text = str(text)

        except:

            text = ''

    return text



def score(text='', verbose=0):

    text = robustify(text)

    for dot in string.punctuation:

        text = text.replace(dot,'')

    tokenized_text = word_tokenize(text)

    if verbose:

        print(f'Tokenized text : {tokenized_text}')

#     stemmed_text = [ps.stem(x) for x in tokenized_text]

#     print(f'Stemmed text : {stemmed_text}')

#     tags = pos_tag(stemmed_text)

    tags = pos_tag(tokenized_text)

    senti_val = [(x.lower(), get_sentiment(x.lower(),y, verbose)) for (x,y) in tags]

    senti_val = list(filter(lambda x : len(x[1])>0, senti_val))

    return senti_val
score('that`s great!! weee!! visitors!', verbose=1)
score('happy bday!', verbose=1)
score('Recession hit Veronique Branquinho', verbose=1)
test['senti_val'] = test['text'].apply(score)
test.loc[test.sentiment != 'neutral'].head()
def treat_senti_val(sentiment, senti_val):

    if sentiment == 'neutral':

        return []

    sent = 0 if sentiment=='positive' else 1

    return [(t[0], t[1][sent]) for t in senti_val] # if t[1][sent]>0

test['senti_val'] = test.apply(lambda df: treat_senti_val(df.sentiment, df.senti_val), axis=1)

test.head(20)
def get_selected_text(text, senti_val):

    if len(senti_val)==0:

        return text

    else:

        return ' '.join([t[0] for t in senti_val if t[1]>0.1])

test['selected_text'] = test.apply(lambda df:get_selected_text(df['text'],df['senti_val']), axis=1)

test.head()
for i, row in test.loc[test.sentiment!='neutral'].iterrows():

    sample_submission.loc[sample_submission.textID==row['textID'], 'selected_text'] = row['selected_text']
sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)