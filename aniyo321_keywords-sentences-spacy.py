# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## Implementing KeyWord Extraction



from collections import OrderedDict

import numpy as np

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

import re

from prettytable import PrettyTable
nlp = spacy.load('en_core_web_sm')
class KeyWordRank():

    ## Extracting Keywords from text



    def __init__(self):

        self.d = 0.85  # damping co-efficinet usually set to 0.85

        self.min_diff = 1e-5 # Threshold for convergence

        self.steps = 10 # number of iteration steps

        self.node_weight = None # save keywords and node_weight



    def imp_stopwords(self, stopwords):

        ## Set the stop_words

        for word in STOP_WORDS.union(set(stopwords)):

            lexeme = nlp.vocab[word]

            lexeme.is_stop = True



    def sent_seg(self, doc, candidate_pos, lower):

        ## Store the words in candidate position

        sentences = []

        for sent in doc.sents:

            sel_words = []

            for token in sent:

                # Store Words only with candidate POS tag

                if token.pos_ in candidate_pos and token.is_stop is False:

                    if lower is True:

                        sel_words.append(token.text.lower())

                    else:

                        sel_words.append(token.text)

            sentences.append(sel_words)

        return sentences



    def get_vocab(self, sentences):

        ## Get all tokens

        vocab = OrderedDict()

        i = 0

        for sentence in sentences:

            for word in sentence:

                if word not in vocab:

                    vocab[word] = i

                    i += 1

        return vocab



    def get_tok_pairs(self, window_size, sentences):

        ## Get all token pairs from windows in sentences

        tok_pairs =  list()

        for sentence in sentences:

            for i, word in enumerate(sentence):

                for j in range(i+1, i+window_size):

                    if j >= len(sentence):

                        break

                    pair = (word, sentence[j])

                    if pair not in tok_pairs:

                        tok_pairs.append(pair)

        return tok_pairs



    def make_symmetrize(self, a):

        return a + a.T - np.diag(a.diagonal())



    def get_mat(self, vocab, tok_pairs):

        ## Get the normalized Matrix

        # Build Matrix

        vocab_size = len(vocab)

        g = np.zeros((vocab_size, vocab_size), dtype='float')

        for word1, word2 in tok_pairs:

            i, j = vocab[word1], vocab[word2]

            g[i][j] = 1



        # Symmetric Matrix

        g = self.make_symmetrize(g)



        # Norm matrix by column

        norm = np.sum(g, axis = 0)

        g_norm = np.divide(g, norm, where=norm!=0)



        return g_norm





    def get_keyword(self, text, number=10):

        ## Get the top Keywords

        

        x = PrettyTable() #Generate the Table

        x.field_names = ["Word(#)", "Documents", "Sentences containing the word" ] #Column names

        x.align["Word(#)"] = "l"

        x.align["Documents"] = "l"

        x.align["Sentences containing the word"] = "l"

        Dict = {} # Initialize the dictionary

        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse = True))

        for i, (key, value) in enumerate(node_weight.items()):

            #print(key + '-' + str(value))

            Dict[key] = ([sentence + '.' for sentence in text.split('.') if key in sentence])

            x.add_row([key,"x,y,z",Dict[key]])

            if i > number:

                print(x)

                break



    def analyze(self, text,

                candidate_pos=['NOUN', 'PROPN'],

                window_size=4, lower=False, stopwords=list()):





                ## Function to Analyze text



                ## Set all stop Words



                self.imp_stopwords(stopwords)



                # Parse text by spacy

                doc = nlp(text)



                ## Filter sentences

                sentences = self.sent_seg(doc, candidate_pos, lower)



                ## Build Vocabulary

                vocab = self.get_vocab(sentences)



                ## Get token pairs from windows

                tok_pairs = self.get_tok_pairs(window_size, sentences)



                ## Get normalized Matrix

                g = self.get_mat(vocab,tok_pairs)



                ## Initialization for weight

                pr = np.array([1] * len(vocab))



                # iteration

                previous = 0

                for epoch in range(self.steps):

                    pr = (1-self.d) + self.d * np.dot(g, pr)

                    if abs(previous - sum(pr)) < self.min_diff:

                        break

                    else:

                        previous = sum(pr)



                # Get weight for each node

                node_weight = dict()

                for word, index in vocab.items():

                    node_weight[word] = pr[index]



                self.node_weight = node_weight
text = "Fusing dyes to antibodies or inserting genes coding for fluorescent proteins into the DNA of living cells can help scientists pick out the location of organelles, cytoskeletal elements, and other subcellular structures from otherwise impenetrable microscopy images. But this technique has its drawbacks. There are limits to the number of fluorescent tags that can be introduced into a cell, and side effects such as photo­toxicity—damage caused by repeated exposure to light—can hinder researchers’ ability to conduct live cell imaging."

tr4w = KeyWordRank()

tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=5, lower=False)

tr4w.get_keyword(text,5)