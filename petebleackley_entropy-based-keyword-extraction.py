# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib inline

import pandas

import numpy

import gensim

import nltk

import seaborn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
alice = open('/kaggle/input/alice-wonderland-dataset/alice_in_wonderland.txt').read()

keywords = pandas.Series(dict(gensim.summarization.mz_entropy.mz_keywords(alice,scores=True)))

keywords.nlargest(20).plot.barh()
keywords2 = pandas.Series(dict(gensim.summarization.mz_entropy.mz_keywords(alice,weighted=False,scores=True)))

keywords2.nlargest(20).plot.barh()
words = list(gensim.summarization.textcleaner.tokenize_by_word(alice))

n_blocks = len(words)//1024 + 1



frequencies = pandas.DataFrame(numpy.zeros((n_blocks,keywords.size)),

                               columns=keywords.index)

for (i,word) in enumerate(words):

    if word in keywords.index:

        frequencies.loc[i//1024,word]+=1

frequencies/=frequencies.sum(axis=0)

seaborn.heatmap(frequencies.loc[:,keywords.nlargest(20).index])
seaborn.heatmap(frequencies.loc[:,keywords2.nlargest(20).index])
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def score_sentence(sentence):

    return sum((keywords2.get(word,0)

               for word in gensim.summarization.textcleaner.tokenize_by_word(sentence)))



sentences = pandas.Series({sentence:score_sentence(sentence)

                          for sentence in sent_detector.tokenize(alice)})

sentences.nlargest(10)
def weighted_score_sentence(sentence):

    return sum((keywords.get(word,0)

               for word in gensim.summarization.textcleaner.tokenize_by_word(sentence)))



weighted_sentences = pandas.Series({sentence:weighted_score_sentence(sentence)

                                    for sentence in sent_detector.tokenize(alice)})

weighted_sentences.nlargest(10)
sentence_lengths = pandas.Series({sentence:len(list(gensim.utils.tokenize(sentence)))

                                 for sentence in sent_detector.tokenize(alice)})

score_vs_length = pandas.DataFrame({'length':sentence_lengths,

                                   'weighted':weighted_sentences,

                                   'unweighted':sentences})

score_vs_length.plot.scatter(x='length',

                            y='unweighted')
score_vs_length.plot.scatter(x='length',

                            y='weighted')
normalized_unweighted = score_vs_length['unweighted']/score_vs_length['length']

normalized_unweighted.nlargest(10)
normalized_weighted = score_vs_length['weighted']/score_vs_length['length']

normalized_weighted.nlargest(10)
root_normalized_unweighted = score_vs_length['unweighted']/score_vs_length['length'].apply(numpy.sqrt)

root_normalized_unweighted.nlargest(10)
root_normalized_weighted = score_vs_length['weighted']/score_vs_length['length'].apply(numpy.sqrt)

root_normalized_weighted.nlargest(10)