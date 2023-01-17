import pandas as pd

from collections import Counter

import tqdm

import regex as re

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

import functools

import sys

from __future__ import division # for python2 compatability

import nltk

from nltk.corpus import stopwords

import nltk

from nltk.corpus import stopwords

#import pymorphy2

import gensim

from gensim.models.keyedvectors import KeyedVectors



%matplotlib inline

import matplotlib.pyplot as plt



plt.style.use('ggplot')



plt.rcParams['figure.figsize'] = (10, 5)
dftrain = pd.read_csv("../input/train_task1_latest.csv", encoding='utf-8')

dftest = pd.read_csv("../input/sdsj_A_test.csv", encoding='utf-8')
dftrain.head()
counter_paragraph = Counter()

uniq_paragraphs = dftrain.paragraph.unique()

for paragraph in uniq_paragraphs:

    counter_paragraph.update(text_to_wordlist(paragraph, remove_stopwords=False, normalize=False))
print('Всего строк: {}, из них {}({}%) строк 1го класса, и, соотвественно {} строк 2го класса.'.\

      format(dftrain.shape[0], dftrain[ dftrain['target'] == 1 ].shape[0],\

            round((dftrain[ dftrain['target'] == 1 ].shape[0]/dftrain.shape[0])*100, 2),\

            dftrain[ dftrain['target'] == 0 ].shape[0]))

print('В таблице следующие колонки [{}]:'.format(dftrain.shape[1]))

for i in dftrain.columns.values:

    print('- ' + i)
class TaskA():

    

    def __init__(self, w2v_fpath = "all.norm-sz100-w10-cb0-it1-min100.w2v"):

        # self.morph_analyzer = pymorphy2.MorphAnalyzer()

        # self.w2v = KeyedVectors.load_word2vec_format(w2v_fpath, binary=True, unicode_errors='ignore')

        # self.w2v.init_sims(replace=True)

        self.stemmer = nltk.stem.snowball.RussianStemmer(ignore_stopwords=False)

        self.dftrain = pd.read_csv("../input/train_task1_latest.csv", encoding='utf-8')

        self.dftest = pd.read_csv("../input/sdsj_A_test.csv", encoding='utf-8')

    

    def train(self):

        for paragraph in dftrain.paragraph.unique():

            self.ounter_paragraph.update(self.text_to_wordlist(paragraph, remove_stopwords=False, normalize=False))

        for question in dftrain.question.unique():

            self.counter_question.update(self.text_to_wordlist(question, remove_stopwords=False, normalize=False))

        paragraph_dic = dict.fromkeys(dftrain.paragraph_id.values)

        #dftrain['paragraph_words']

        #dftrain['question_words']

        return self.counter_paragraph

    

    def dftest(dict=True):

        return self.dftest

    

    @staticmethod

    def text_to_wordlist(text, remove_stopwords=False, normalize=False):

            words = re.sub("[^a-яА-Я]", " ", re.sub('[Ёё]', 'е', text)).lower().split()

            if remove_stopwords:

                stops = set(stopwords.words("russian"))

                words = [w for w in words if not w in stops]

            if normalize:

                words = [self.stemmer.stem(w) for w in words]

            return(words)

    

    @staticmethod

    def text_to_sentences(text, remove_stopwords=False):

        raw_sentences = nltk.sent_tokenize(text)

        sentences = []

        for raw_sentence in raw_sentences:

            if len(raw_sentence) > 0:

                sentences.append( TaskA.text_to_wordlist( raw_sentence, remove_stopwords ))

        return sentences

    

    def calculate_idfs(self, data):

        counter_paragraph = Counter()

        uniq_paragraphs = data['paragraph'].unique()

        for paragraph in uniq_paragraphs:

            counter_paragraph.update(text_to_wordlist(paragraph, remove_stopwords=False, normalize=False))

        num_docs = uniq_paragraphs.shape[0]

        idfs = {}

        for word in counter_paragraph:

            idfs[word] = ln(num_docs / counter_paragraph[word])

        return idfs

    

    def make_features(self, data):

        for name, df in [('train', dftrain), ('test', dftest)]:

            for index, row in tqdm.tqdm(df.iterrows(), \

                               total=df.shape[0],\

                               desc="build features for " + name):

                question = TaskA.text_to_wordlist((row.question))

                paragraph = TaskA.text_to_wordlist((row.paragraph))

                df.loc[index, 'len_paragraph'] = len(paragraph)

                df.loc[index, 'len_question'] = len(question)

                df.loc[index, 'len_intersection'] = len(paragraph & question)

                df.loc[index, 'idf_question'] = np.sum([idfs.get(word, 0.0) for word in question])

                df.loc[index, 'idf_paragraph'] = np.sum([idfs.get(word, 0.0) for word in paragraph])

                df.loc[index, 'idf_intersection'] = np.sum([idfs.get(word, 0.0) for word in paragraph & question])

                

    def prediction(dftrain, dftest, columns):

        columns = ['len_paragraph', 'len_question', 'len_intersection', 'idf_question', 'idf_paragraph', 'idf_intersection']

        model = GradientBoostingClassifier().fit(dftrain[columns], dftrain['target'])

        dftest['prediction'] = model.predict_proba(dftest[columns])[:, 1]

        return dftest

                                                   

    def prediction_to_csv(dftest):

        dftest[['paragraph_id', 'question_id', 'prediction']].to_csv("prediction1.csv", index=False)
f = TaskA()
dict.fromkeys(dftrain.paragraph_id.values)