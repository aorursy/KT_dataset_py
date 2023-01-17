#!python -m spacy download en_core_web_lg
import numpy as np

import pandas as pd

import matplotlib.pylab as plt

import random

import seaborn

import re

import string

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

from sklearn import metrics

from sklearn.base import TransformerMixin

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train_df = pd.read_csv('../input/nlp-getting-started/train.csv')

train_df.info()
train_df.head()
train_df.target.value_counts()
def remove_url(txt):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',txt)



train_df.text = train_df.text.apply(lambda x: remove_url(x))
X_train, X_valid, y_train, y_valid = train_test_split(train_df.text.tolist(),

                                                      train_df.target.tolist(),

                                                      test_size=0.1)
nlp = spacy.load('en_core_web_lg')

nlp.pipeline
TRAIN_DATA = [('OMG! Earthquake in Tehran!', {'cats': {'POSITIVE': 1}}),

              ('Has an accident changed your life?', {'cats': {'POSITIVE': 0}})]



def parse_train_data(docs, labels) :    

    train_data = []

    for doc, label in zip(docs, labels) :

        S = {'cats': {'POSITIVE': label}}

        train_data.append((doc.text, S))

        

    return train_data



docs = [doc for doc in nlp.pipe(X_train)]

train_data = parse_train_data(docs, y_train)



random.choices(train_data, k=5)
if 'textcat' not in nlp.pipe_names:

    textcat = nlp.create_pipe("textcat")

    nlp.add_pipe(textcat, last=True) 

else:

    textcat = nlp.get_pipe("textcat")



textcat.add_label('POSITIVE')



nlp.pipeline
fixed_pipes = [pipe for pipe in nlp.pipe_names if pipe!='textcat']

fixed_pipes
## https://spacy.io/usage/training#textcat

import datetime as dt

from spacy.util import minibatch, compounding



def evaluate(tokenizer, textcat, texts, cats):

    docs = (tokenizer(text) for text in texts)

    tp = 0.0  # True positives

    fp = 1e-8  # False positives

    fn = 1e-8  # False negatives

    tn = 0.0  # True negatives

    for ii, doc in enumerate(docs):

        score, _ = textcat.predict([doc])

        if score[0][0] >= 0.5 and cats[ii] == 1:

            tp += 1.0

        elif score[0][0] >= 0.5 and cats[ii] == 0:

            fp += 1.0

        elif score[0][0] < 0.5 and cats[ii] == 0:

            tn += 1

        elif score[0][0] < 0.5 and cats[ii] == 1:

            fn += 1

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    if (precision + recall) == 0:

        f_score = 0.0

    else:

        f_score = 2 * (precision * recall) / (precision + recall)

    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}





## Only train the textcat pipe

print("Loss \t P\t R\t F1")

with nlp.disable_pipes(*fixed_pipes):

    optimizer = nlp.begin_training()

    batch_sizes = compounding(128, 256, 512)

    for itn in range(50) :

        random.shuffle(train_data)

        losses = {}

        minibatches = minibatch(train_data, size=batch_sizes)

        for batch in minibatches :

            txts, annotations = zip(*batch)

            nlp.update(txts, 

                       annotations, 

                       sgd=optimizer, 

                       drop=0.1,

                       losses=losses)

        with textcat.model.use_params(optimizer.averages):

            # evaluate on the valid data split off in load_data()

            scores = evaluate(nlp.tokenizer, textcat, X_valid, y_valid)

        print("{0:.5f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(

                    losses["textcat"],

                    scores["textcat_p"],

                    scores["textcat_r"],

                    scores["textcat_f"]))
# print mislabelled examples

docs = [nlp.tokenizer(text) for text in X_valid]        

for ii in range(50):

    score, _ = textcat.predict([docs[ii]])

    if ((score[0][0] >= 0.5 and y_valid[ii] == 0) or

        (score[0][0] < 0.5 and y_valid[ii] == 1)) :

        print('pred: ' + str(score[0][0]) + 

              ', true: ' + str(y_valid[ii]) + 

               ' --> ', X_valid[ii] + '\n')
## 4

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')

test_df.text = test_df.text.apply(lambda x: remove_url(x))

test_df['target'] = np.nan



X_test = test_df.text.tolist()

docs = [nlp.tokenizer(text) for text in X_test]        

for ii, doc in enumerate(docs[:10]) :

    score, _ = textcat.predict([docs[ii]])

    test_df.target[ii] = (0 if score[0][0] < 0.5 else 1)



test_df.head(10)
sub_df = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub_df.target = test_df.target

sub_df.head()

sub_df.to_csv("sub-spacy.csv", index=False, header=True)