import numpy as np # linear algebra

import pandas as pd # data processing

import os

data_paths = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data_paths[filename] = os.path.join(dirname, filename)

        

print(data_paths)
!pip -q install sklearn-crfsuite seqeval
ner_df = pd.read_csv(data_paths['ner_dataset.csv'], encoding = 'unicode_escape')

ner_df.fillna(method = 'ffill', inplace = True)

ner_df.rename(columns = {'Sentence #':'SentId'}, inplace = True)

print(ner_df.shape, ner_df.columns)

ner_df['SentId'] = ner_df['SentId'].apply(lambda x:x.split()[-1]).astype(int)

ner_df.head()
agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values,s["POS"].values, s["Tag"].values)]

grouped = ner_df.groupby('SentId').apply(agg_func)

sentences = [s for s in grouped]
val_end = int(47959 * 0.84)

train_end = int(val_end*0.8)

train_end, val_end
train_sents = sentences[:train_end]

val_sents = sentences[train_end:val_end]

test_sents = sentences[val_end:]

len(train_sents), len(val_sents), len(test_sents)
from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize 

from nltk.corpus import stopwords

eng_stops = stopwords.words('english')

ps = PorterStemmer() 



def is_stopword(word):

    r = set([word]) & set(eng_stops)

    if len(r):

        return True

    return False

    

def word2features(sent, i):

    word = sent[i][0]

    postag = sent[i][1]



    features = {

        'bias': 1.0,

        'index':i,

        'is_stopword':is_stopword(word.lower()), 

        'word.stem':ps.stem(word.lower()), 

        'word.lower': word.lower(),

        'word[-3:]': word[-3:],

        'word[-2:]': word[-2:],

        'word.isupper': word.isupper(),

        'word.istitle': word.istitle(),

        'word.isdigit': word.isdigit(),

        'postag': postag,

        'postag[:2]': postag[:2],

    }

    if i > 0:

        word1 = sent[i-1][0]

        postag1 = sent[i-1][1]

        features.update({

            '-1:index':i-1,

            '-1:is_stopword':is_stopword(word1.lower()), 

            '-1:word.stem':ps.stem(word1.lower()), 

            '-1:word.lower': word1.lower(),

            '-1:word.istitle': word1.istitle(),

            '-1:word.isupper': word1.isupper(),

            '-1:word.isdigit': word1.isdigit(),

            '-1:postag': postag1,

            '-1:postag[:2]': postag1[:2],

        })

    else:

        features['BOS'] = True



    if i < len(sent)-1:

        word1 = sent[i+1][0]

        postag1 = sent[i+1][1]

        features.update({

            '+1:index':i+1, 

            '+1:is_stopword':is_stopword(word1.lower()), 

            '+1:word.stem':ps.stem(word1.lower()), 

            '+1:word.lower': word1.lower(),

            '+1:word.istitle': word1.istitle(),

            '+1:word.isupper': word1.isupper(),

            '+1:word.isdigit': word1.isdigit(),

            '+1:postag': postag1,

            '+1:postag[:2]': postag1[:2],

        })

    else:

        features['EOS'] = True



    return features



def sent2features(sent):

    return [word2features(sent, i) for i in range(len(sent))]



def sent2labels(sent):

    return [label for token, postag, label in sent]



def sent2tokens(sent):

    return [token for token, postag, label in sent]
X_train = [sent2features(s) for s in train_sents]

y_train = [sent2labels(s) for s in train_sents]



X_val = [sent2features(s) for s in val_sents]

y_val = [sent2labels(s) for s in val_sents]



X_test = [sent2features(s) for s in test_sents]

y_test = [sent2labels(s) for s in test_sents]
import sklearn_crfsuite

from seqeval.metrics import f1_score, classification_report



crf = sklearn_crfsuite.CRF(algorithm='lbfgs',

                           c1=0.3, c2=0.05,

                           max_iterations=250,

                           all_possible_transitions=True)

crf.fit(X_train, y_train)



print('Performance on TRAIN...')

y_train_pred = crf.predict(X_train)

print(f1_score(y_train, y_train_pred))

print(classification_report(y_train, y_train_pred))
print('Evaluation on DEV...')

y_val_pred = crf.predict(X_val)

print('f1-score:', f1_score(y_val, y_val_pred))

print(classification_report(y_val, y_val_pred))
print('Evaluation on TEST...')

y_test_pred = crf.predict(X_test)

print('f1-score:',f1_score(y_test, y_test_pred))

print(classification_report(y_test, y_test_pred))