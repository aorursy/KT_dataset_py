from __future__ import division

import gensim

import numpy as np

from sklearn.decomposition import PCA

import gensim.models.word2vec

from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

import itertools

from nltk.tokenize import word_tokenize

import re

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

from string import punctuation

import csv

print("All Libraries Successfully imported")
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):

    # Clean the text, with the option to remove stopwords and to stem words.

    

    # Convert words to lower case and split them

    text = text.lower().split()



    # Optionally, remove stop words

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        text = [w for w in text if not w in stops]

    

    text = " ".join(text)



    # Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    

    # Optionally, shorten words to their stems

    if stem_words:

        text = text.split()

        stemmer = SnowballStemmer('english')

        stemmed_words = [stemmer.stem(word) for word in text]

        text = " ".join(stemmed_words)

    

    # Return a list of words

    return(text)
#Initializing word2vec



def gensim_load_vec(path="GoogleNews-vectors-negative300.bin"):

    #use gensim_emb.wv.index2word if used this way to load vectors

    #gensim_emb = gensim.models.word2vec.Word2Vec.load(path)

    gensim_emb =  gensim.models.KeyedVectors.load_word2vec_format(path, binary=True,limit=500000)

    vocab = gensim_emb.index2word

    vec = gensim_emb.syn0

    shape = gensim_emb.syn0.shape

    return gensim_emb, vec, shape, vocab
def map_word_frequency(document):

    return Counter(itertools.chain(*document))


def sentence2vec(tokenised_sentence_list, embedding_size, word_emb_model, a = 1e-3):

    sentence_vecs = []

    try:

        word_counts = map_word_frequency(tokenised_sentence_list)

        sentence_set=[]

        for sentence in tokenised_sentence_list:

            vs = np.zeros(embedding_size)

            sentence_length = len(sentence)

            for word in sentence:

                a_value = a / (a + word_counts[word]) # smooth inverse frequency, SIF

            try:

                vs = np.add(vs, np.multiply(a_value, word_emb_model[word])) # vs += sif * word_vector

            except Exception as e:

                pass

            vs = np.divide(vs, sentence_length) # weighted average

            sentence_set.append(vs)

    except Exception as e:

           print("Exception in sentence embedding")

    return sentence_set
#Testing with one pair



text1 = "How can I be good geologist"

text2 = "should I do be great geologist"

token1 = text_to_wordlist(text1)

token2 = text_to_wordlist(text2)

token1 = word_tokenize(token1)

token2 = word_tokenize(token2)

sent_list=[]

sent_list.append(token1)

sent_list.append(token2)

gensim_emb, vec, shape, vocab = gensim_load_vec()
sent_emb = sentence2vec(sent_list,300,gensim_emb)
score=float(cosine_similarity([sent_emb[0]],[sent_emb[1]]))

print(score)