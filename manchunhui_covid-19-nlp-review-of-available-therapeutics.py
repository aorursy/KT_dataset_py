!pip install langdetect

!pip install num2words
import numpy as np

import pandas as pd

import gensim

import nltk

import string

import spacy

import langdetect

import networkx as nx

import warnings

import math

import matplotlib.pyplot as plt

import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)



pd.set_option('display.max_colwidth', 0)



from spacy import displacy

from spacy.lang.en import English

from spacy.lang.en.stop_words import STOP_WORDS



from collections import Counter



from num2words import num2words



from nltk import word_tokenize          

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize

lemmatizer = WordNetLemmatizer()



from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction import text

stop = text.ENGLISH_STOP_WORDS.union(["book"])



from langdetect import detect



from gensim import models

from gensim.test.utils import datapath, get_tmpfile

from gensim.models import KeyedVectors



from IPython.core.display import display, HTML



from time import time  # To time our operations



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.decomposition import PCA as sPCA

from sklearn import manifold #MSD



def show_closest_2d(vecs,word1,word2,word3,word4,n):

    tops1 = vecs.similar_by_word(word1, topn=n, restrict_vocab=None)

    tops2 = vecs.similar_by_word(word2, topn=n, restrict_vocab=None)

    tops3 = vecs.similar_by_word(word3, topn=n, restrict_vocab=None)

    tops4 = vecs.similar_by_word(word4, topn=n, restrict_vocab=None)

    

    #display(HTML("<b>%d words most similar to '%s' (%s)</b>" % (n,word, method)))

   

    items1 = [word1] + [x[0] for x in tops1]

    items2 = [word2] + [x[0] for x in tops2]

    items3 = [word3] + [x[0] for x in tops3]

    items4 = [word4] + [x[0] for x in tops4]

    

    wvecs1 = np.array([vecs.word_vec(wd, use_norm=True) for wd in items1])

    wvecs2 = np.array([vecs.word_vec(wd, use_norm=True) for wd in items2])

    wvecs3 = np.array([vecs.word_vec(wd, use_norm=True) for wd in items3])

    wvecs4 = np.array([vecs.word_vec(wd, use_norm=True) for wd in items4])



    dists1 = np.zeros((len(items1), len(items1)))

    dists2 = np.zeros((len(items2), len(items2)))

    dists3 = np.zeros((len(items3), len(items3)))

    dists4 = np.zeros((len(items4), len(items4)))

    

    for i,item1 in enumerate(items1):

        for j,item2 in enumerate(items1):

            dists1[i][j] = dists1[j][i] = vecs.distance(item1,item2)

            

    for i,item1 in enumerate(items2):

        for j,item2 in enumerate(items2):

            dists2[i][j] = dists2[j][i] = vecs.distance(item1,item2)

            

    for i,item1 in enumerate(items3):

        for j,item2 in enumerate(items3):

            dists3[i][j] = dists3[j][i] = vecs.distance(item1,item2)

            

    for i,item1 in enumerate(items4):

        for j,item2 in enumerate(items4):

            dists4[i][j] = dists4[j][i] = vecs.distance(item1,item2)

        

    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=0, 

                       dissimilarity="precomputed", n_jobs=1)

    

    coords1 = mds.fit(dists1).embedding_

    coords2 = mds.fit(dists2).embedding_

    coords3 = mds.fit(dists3).embedding_

    coords4 = mds.fit(dists4).embedding_

    

    plt.figure(num=None, figsize=(8, 4), dpi=1200, facecolor='w', edgecolor='k')

    

    plt.subplot(221)

    plt.tick_params(

        axis='both',          

        which='both',      

        bottom=False,      

        left=False,         

        labelbottom=False,

        labelleft=False)



    lim1 = max([abs(x) for x in coords1[:,0] + coords1[:,1]])

    plt.xlim([-lim1,lim1])

    plt.ylim([-lim1,lim1])

    

    plt.scatter(coords1[0:,0], coords1[0:,1], color='darkgrey', s=3)

    plt.scatter(coords1[4:5,0], coords1[4:5,1], color='darkviolet', s=3)

    

    for item, x, y in zip(items1[2:], coords1[2:,0], coords1[2:,1]):

        plt.annotate(item, xy=(x, y), xytext=(-1, 1), textcoords='offset points', 

                     ha='right', va='bottom', color='grey', fontsize=6)



    x0=coords1[0,0]

    y0=coords1[0,1]

    plt.annotate(word1, xy=(x0, y0), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='grey', fontsize=6)

    

    x1=coords1[1,0]

    y1=coords1[1,1]

    plt.annotate(items1[1] , xy=(x1, y1), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='grey', fontsize=6)

    

    x4=coords1[4,0]

    y4=coords1[4,1]

    plt.annotate(items1[4] , xy=(x4, y4), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='darkviolet', fontsize=6)

    

    ax = plt.gca()

    

    circle = plt.Circle((x0, y0), 0.1, color='black', fill=False)

    ax.add_artist(circle)

    

    plt.subplot(222)

    plt.tick_params(

        axis='both',

        which='both',

        bottom=False,      

        left=False,         

        labelbottom=False,

        labelleft=False)

    

    lim2 = max([abs(x) for x in coords2[:,0] + coords2[:,1]])

    plt.xlim([-lim2,lim2])

    plt.ylim([-lim2,lim2])

    

    plt.scatter(coords2[0:,0], coords2[0:,1], color='grey', s=3)

    plt.scatter(coords2[5:6,0], coords2[5:6,1], color='blue', s=3)

    plt.scatter(coords2[6:7,0], coords2[6:7,1], color='darkviolet', s=3)

    plt.scatter(coords2[7:8,0], coords2[7:8,1], color='blue', s=3)

    

    for item, x, y in zip(items2[2:], coords2[2:,0], coords2[2:,1]):

        plt.annotate(item, xy=(x, y), xytext=(-1, 1), textcoords='offset points', 

                     ha='right', va='bottom', color='grey', fontsize=6)



    

    x0=coords2[0,0]

    y0=coords2[0,1]

    plt.annotate(word2, xy=(x0, y0), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='grey', fontsize=6)

    

    x1=coords2[1,0]

    y1=coords2[1,1]

    plt.annotate(items2[1], xy=(x1, y1), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='grey', fontsize=6)

    

    x5=coords2[5,0]

    y5=coords2[5,1]

    plt.annotate(items2[5] , xy=(x5, y5), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='blue', fontsize=6)

    

    x6=coords2[6,0]

    y6=coords2[6,1]

    plt.annotate(items2[6] , xy=(x6, y6), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='darkviolet', fontsize=6)

    

    x7=coords2[7,0]

    y7=coords2[7,1]

    plt.annotate(items2[7] , xy=(x7, y7), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='blue', fontsize=6)

    

    ax = plt.gca()

    

    circle = plt.Circle((x0, y0), 0.1, color='black', fill=False)

    ax.add_artist(circle)

    

    plt.subplot(223)

    plt.tick_params(

        axis='both',

        which='both',

        bottom=False,      

        left=False,         

        labelbottom=False,

        labelleft=False)

    

    lim3 = max([abs(x) for x in coords3[:,0] + coords3[:,1]])

    plt.xlim([-lim3,lim3])

    plt.ylim([-lim3,lim3])

    

    plt.scatter(coords3[0:,0], coords3[0:,1], color='grey', s=3)

    plt.scatter(coords3[0:1,0], coords3[0:1,1], color='forestgreen', s=3)

    plt.scatter(coords3[1:2,0], coords3[1:2,1], color='blue', s=3)

    plt.scatter(coords3[2:3,0], coords3[2:3,1], color='darkviolet', s=3)

    plt.scatter(coords3[3:4,0], coords3[3:4,1], color='darkviolet', s=3)

    plt.scatter(coords3[4:5,0], coords3[4:5,1], color='blue', s=3)

    plt.scatter(coords3[5:6,0], coords3[5:6,1], color='forestgreen', s=3)

    plt.scatter(coords3[8:9,0], coords3[8:9,1], color='darkorange', s=3)

    plt.scatter(coords3[10:,0], coords3[10:,1], color='black', s=3)

    

    for item, x, y in zip(items3[0:], coords3[0:,0], coords3[0:,1]):

        plt.annotate(item, xy=(x, y), xytext=(-1, 1), textcoords='offset points', 

                     ha='right', va='bottom', color='grey', fontsize=6)



    x0=coords3[0,0]

    y0=coords3[0,1]

    plt.annotate(word3, xy=(x0, y0), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='forestgreen', fontsize=6)

    

    x1=coords3[1,0]

    y1=coords3[1,1]

    plt.annotate(items3[1] , xy=(x1, y1), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='blue', fontsize=6)

    

    x2=coords3[2,0]

    y2=coords3[2,1]

    plt.annotate(items3[2] , xy=(x2, y2), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='darkviolet', fontsize=6)

    

    x3=coords3[3,0]

    y3=coords3[3,1]

    plt.annotate(items3[3] , xy=(x3, y3), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='darkviolet', fontsize=6)

    

    x4=coords3[4,0]

    y4=coords3[4,1]

    plt.annotate(items3[4] , xy=(x4, y4), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='blue', fontsize=6)

    

    x5=coords3[5,0]

    y5=coords3[5,1]

    plt.annotate(items3[5] , xy=(x5, y5), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='forestgreen', fontsize=6)

    

    x8=coords3[8,0]

    y8=coords3[8,1]

    plt.annotate(items3[8] , xy=(x8, y8), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='darkorange', fontsize=6)

    

    x10=coords3[10,0]

    y10=coords3[10,1]

    plt.annotate(items3[10] , xy=(x10, y10), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='black', fontsize=6)

    

    ax = plt.gca()

    circle = plt.Circle((x0, y0), 0.1, color='black', fill=False)

    ax.add_artist(circle)

    

    plt.subplot(224)

    plt.tick_params(

        axis='both',

        which='both',

        bottom=False,      

        left=False,         

        labelbottom=False,

        labelleft=False)

    

    lim4 = max([abs(x) for x in coords4[:,0] + coords4[:,1]])

    plt.xlim([-lim4,lim4])

    plt.ylim([-lim4,lim4])

    

    plt.scatter(coords4[0:,0], coords4[0:,1], color='grey', s=3)

    plt.scatter(coords4[0:1,0], coords4[0:1,1], color='black', s=3)

    plt.scatter(coords4[1:2,0], coords4[1:2,1], color='turquoise', s=3)

    plt.scatter(coords4[2:3,0], coords4[2:3,1], color='black', s=3)

    plt.scatter(coords4[3:4,0], coords4[3:4,1], color='black', s=3)

    plt.scatter(coords4[4:5,0], coords4[4:5,1], color='black', s=3)

    plt.scatter(coords4[5:6,0], coords4[5:6,1], color='red', s=3)

    plt.scatter(coords4[6:7,0], coords4[6:7,1], color='forestgreen', s=3)

    plt.scatter(coords4[7:8,0], coords4[7:8,1], color='brown', s=3)

    plt.scatter(coords4[8:9,0], coords4[8:9,1], color='blue', s=3)

    plt.scatter(coords4[9:10,0], coords4[9:10,1], color='darkviolet', s=3)

    plt.scatter(coords4[10:11,0], coords4[10:11,1], color='forestgreen', s=3)

    

    

    for item, x, y in zip(items4[0:], coords4[0:,0], coords4[0:,1]):

        plt.annotate(item, xy=(x, y), xytext=(-1, 1), textcoords='offset points', 

                     ha='right', va='bottom', color='grey', fontsize=6)



    x0=coords4[0,0]

    y0=coords4[0,1]

    plt.annotate(word4, xy=(x0, y0), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='black', fontsize=6)

    

    x1=coords4[1,0]

    y1=coords4[1,1]

    plt.annotate(items4[1] , xy=(x1, y1), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='turquoise', fontsize=6)

    

    x2=coords4[2,0]

    y2=coords4[2,1]

    plt.annotate(items4[2], xy=(x2, y2), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='black', fontsize=6)

    

    x3=coords4[3,0]

    y3=coords4[3,1]

    plt.annotate(items4[3], xy=(x3, y3), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='black', fontsize=6)

    

    x4=coords4[4,0]

    y4=coords4[4,1]

    plt.annotate(items4[4], xy=(x4, y4), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='black', fontsize=6)

    

    x5=coords4[5,0]

    y5=coords4[5,1]

    plt.annotate(items4[5], xy=(x5, y5), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='red', fontsize=6)

    

    x6=coords4[6,0]

    y6=coords4[6,1]

    plt.annotate(items4[6], xy=(x6, y6), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='forestgreen', fontsize=6)



    x7=coords4[7,0]

    y7=coords4[7,1]

    plt.annotate(items4[7], xy=(x7, y7), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='brown', fontsize=6)

    

    x8=coords4[8,0]

    y8=coords4[8,1]

    plt.annotate(items4[8], xy=(x8, y8), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='blue', fontsize=6)

    

    x9=coords4[9,0]

    y9=coords4[9,1]

    plt.annotate(items4[9], xy=(x9, y9), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='darkviolet', fontsize=6)

    

    x10=coords4[10,0]

    y10=coords4[10,1]

    plt.annotate(items4[10], xy=(x10, y10), xytext=(-1, 1), textcoords='offset points', 

                 ha='right', va='bottom', color='forestgreen', fontsize=6)

    

    ax = plt.gca()

    circle = plt.Circle((x0, y0), 0.1, color='black', fill=False)

    ax.add_artist(circle)

    

    plt.subplots_adjust(wspace=0, hspace=0)

    

    plt.show()
# load the meta data and reduce column count

df=pd.read_csv('/kaggle/input/eg-gensim-covid-sg-v2/metadata_mod.csv', 

               usecols=['title','journal','abstract',

                        'authors','doi','publish_time',

                        'sha','full_text_file','url'],

               encoding='utf-8', low_memory=False)

df.tail(2)
df.describe()
def treat_na(df):

    df = df.fillna('no data')

    return df



def lower_case(df):

    df['abstract_processed'] = df["abstract"].str.lower()

    return df



string_map1 = {

    'μm': 'micromolar',

    '%': '-percent',

    'Œºg' : 'microgram'

}



def special_character_conversion(df):

    for key, value in string_map1.items():

        df['abstract_processed'] = df['abstract_processed'].str.replace(key, value)

    return df



string_map2 = {

    'α': 'alpha',

    'β': 'beta',

    'γ': 'gamma',

    'δ': 'delta',

    'ε': 'epsilon',

    'ζ': 'zeta',

    'η': 'eta',

    'θ': 'theta',

    'ι': 'iota',

    'κ': 'kappa',

    'λ': 'lambda',

    'μ': 'mu',

    'ν': 'nu',

    'ξ': 'xi',

    'ο': 'omicron',

    'π': 'pi',

    'ρ': 'rho',

    'σ': 'sigma',

    'τ': 'tau',

    'υ': 'upsilon',

    'φ': 'phi',

    'χ': 'chi',

    'ψ': 'psi',

    'ω': 'omega'

}



def greek_alphabet_conversion(df):

    for key, value in string_map2.items():

        df['abstract_processed'] = df['abstract_processed'].str.replace(key, value)

    return df



string_map3 = {

    'sars-cov-two': 'sars-cov-2',

    'sars-covtwo': 'sars-cov2',

    'covid-nineteen': 'covid-19',

    'covid 19': 'covid-19',

    'two thousand and nineteen-ncov': '2019-ncov',

    'two thousand and nineteen': '2019',

    'hsv-one': 'hsv-1',

    'hsv 1': 'hsv-1',

    'hsv-two': 'hsv-2',

    'hsv 2': 'hsv-2',

    'hfivenone': 'h5n1',

    'honenone': 'h1n1',

    'hsevennnine': 'h7n9',

    'interferon beta': 'interferon-beta',

    'interferon gamma': 'interferon-gamma',

    'ifn  -alpha' : 'ifn-alpha',

    'ifnalpha': 'ifn-alpha',

    'ifnbeta': 'ifn-beta',

    'ifngamma': 'ifn-gamma',

    'type i interferon': 'type-i-interferon',

    'type i interferons': 'type-i-interferon',

    'type i ifn': 'type-i-ifn',

    'type ii ifn': 'type-ii-ifn',

    'ifn type i': 'ifn-type-i',

    'ifn type ii': 'ifn-type-ii',

    'gamma interferon': 'gamma-interferon',

    'interferon gamma': 'interferon-gamma',

    'twenty-five-hydroxyvitamin d': '25-hydroxyvitamin-d',

    'twenty-five-dihydroxyvitamin d': '25-dihydroxyvitamin-d',

    'one,twenty-five-dihydroxyvitamin d': '25-dihydroxyvitamin-d',

    '25-hydroxyvitamin d': '25-hydroxyvitamin-d',

    '25-dihydroxyvitamin d': '25-dihydroxyvitamin-d',

    '1,25-dihydroxyvitamin d': '25-dihydroxyvitamin-d',

    'twenty-five(oh)dthree': '25-oh-d3',

    '(twenty-five(oh)d)': '25-oh-d',

    'twenty-five(oh)d': '25-oh-d',

    'twenty-fiveohd': '25-oh-d',

    '1,25 ( oh ) 2d': '25-oh-d2',

    'twenty-fiveohd': '25-oh-d',

    '25ohd': '25-oh-d',

    'twenty-five ( oh ) d': '25-oh-d',

    'twenty-five  oh  d' : '25-oh-d',

    'angiotensin-converting enzyme': 'angiotensin-converting-enzyme',

    'angiotensin-converting enzyme two': 'angiotensin-converting-enzyme-two',

    'angiotensin-converting-enzyme two': 'angiotensin-converting-enzyme-two',

    'angiotensin-converting enzyme ii': 'angiotensin-converting-enzyme-ii',

    'angiotensin-converting-enzyme ii': 'angiotensin-converting-enzyme-ii',

    'angiotensin-i converting enzyme' : 'angiotensin-i-converting-enzyme',

    'angiotensin-ii converting enzyme' : 'angiotensin-ii-converting-enzyme',

    'rna polymerase': 'rna-polymerase',

    'rna dependent': 'rna-dependent',

    'lopinavir-ritonavir': 'lopinavir ritonavir',

    'gs 5734': 'gs-5734',

    'acetwo': 'ace2',

    'interferon-œ±2b' : 'interferon-alpha2b',

    'î²' : 'beta',

    'ifn-œ±2b' : 'interferon-alpha2b',

    'ifn  -œ±2b' : 'interferon-alpha2b',

    'rifn-œ±2a' : 'rifn-alpha2b',

    'rifn-œ±2b' : 'rifn-beta2b',

    'interferon  ifn  -œ±2b': 'interferon-alpha2b',

    'IFN-Œ±2b' : 'IFN-a2b',

    'TGF-Œ≤' : 'TGF-beta',

    'TNFŒ±' : 'TNFalpha',

    'Masson‚Äôs' : "Masson's",

    '3‚Äì7' : '3-7',

    '3‚Äì14' : '3-14',

    '6‚Äì8' : '6-8',

    '7‚Äì9' : '7-9',

    'product‚Äô' : "product'",

    'day‚Ä≤s' : " day's",

    'its‚Ä≤' : "its'"

}



def terms_standardisation1(df):

    for key, value in string_map3.items():

        df['abstract_processed'] = df['abstract_processed'].str.replace(key, value)

    return df



string_map4 = {

    'vitamin c': 'ascorbic-acid',

    'vitamin d  three' : '25-hydroxyvitamin-d',

    'vitamin d-binding': 'vitamin-d binding',

    'vitamin d-': 'vitamin-d ',

    'vitamin d': 'vitamin-d',

    'vit d': 'vitamin-d',

    '1,25-  oh   two  -vitamin-d' : '25-hydroxyvitamin-d'

}



def terms_standardisation2(df):

    for key, value in string_map4.items():

        df['abstract_processed'] = df['abstract_processed'].str.replace(key, value)

    return df



def remove_punctuation1(df):

    symbols = "()[\]" # Not included*-/

    for i in symbols:

        df['abstract_processed'] = df['abstract_processed'].str.replace(i, '')

    return df



def remove_punctuation2(df):

    symbols = "/" 

    for i in symbols:

        df['abstract_processed'] = df['abstract_processed'].str.replace(i, ' ')

    return df



# keep only documents with relation to the subject

disease =['covid',

          'covid-19',

          '-cov2', 

          'cov2',

          'sars-cov-2',

          'sars-cov2',

          'sars',

          '2019-ncov',

          'sars-cov']



pattern = '|'.join(disease)



def search_focus(df):

    df = df[df['abstract_processed'].str.contains(pattern)]

    return df



def remove_duplicates(df):

    df = df.drop_duplicates(subset='title', keep="first")

    return df



def numbers_to_words(data):

    tokens = word_tokenize(str(data))

    new_text = ""

    for w in tokens:

        try:

            w = num2words(int(w))

        except:

            a = 0

        new_text = new_text + " " + w

    return new_text



def convert_numbers(df):

    df['abstract_processed'] = df.apply(lambda row: numbers_to_words(row['abstract_processed']),axis=1)

    return df



def remove_non_en(df):

    df['lang'] = df.apply(lambda row: detect(row['abstract']),axis=1)

    df = df[df['lang'] == 'en']

    return df
def preprocess(data):

    data = treat_na(data)

    data = lower_case(data)

    data = remove_duplicates(data)

    data = search_focus(data)

    data = remove_non_en(data)

    data = special_character_conversion(data)

    data = greek_alphabet_conversion(data)

    data = convert_numbers(data)

    data = terms_standardisation1(data)

    data = remove_punctuation1(data)

    data = terms_standardisation2(data)

    data = remove_punctuation2(data)

    return data



t = time()

df=preprocess(df)

print('Time to preprocess everything: {} mins'.format(round((time() - t) / 60, 2)))
df.describe()
tmp_file = get_tmpfile('/kaggle/input/eg-gensim-covid-sg-v2/gensim-model-20200510.txt')

model = KeyedVectors.load_word2vec_format(tmp_file)

word_vectors = model
word_vectors.similar_by_word("treatments")
show_closest_2d(word_vectors,'treatment','antivirals','lopinavir_ritonavir','hydroxychloroquine',10)
t = time()

# Create our list of punctuation marks

punctuations = string.punctuation



# Create our list of stopwords

nlp = spacy.load('/kaggle/input/eg-gensim-covid-sg-v2/en_gensim_covid_sg_v2')

stop_words = spacy.lang.en.stop_words.STOP_WORDS



# Load English tokenizer, tagger, parser, NER and word vectors

parser = English()



# Creating our tokenizer function

def scispacy_tokenizer(doc):

    # Creating our token object, which is used to create documents with linguistic annotations.

    mytokens = parser(doc)



    # Lemmatizing each token and converting each token into lowercase

    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]



    # Removing stop words and punctuation

    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]



    # return preprocessed list of tokens

    return mytokens



print('Time to preprocess everything: {} mins'.format(round((time() - t) / 60, 2)))
#Calculate tfidf for all literature

t = time()



#Calculate tf

cv = CountVectorizer(tokenizer=scispacy_tokenizer,

                     ngram_range=(1,3),

                     max_df=0.75,

                     min_df=2,

                     stop_words=stop, 

                     token_pattern='[a-z]+\w*')

word_count_vector = cv.fit_transform(df['abstract_processed'])



#Calculate idf

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

tfidf_transformer.fit(word_count_vector)



#Calculate tfidf

count_vector=cv.transform(df['abstract_processed'])

tf_idf_vector=tfidf_transformer.transform(count_vector)



#Create feature names

feature_names = np.array(cv.get_feature_names())



print('Time to preprocess everything: {} mins'.format(round((time() - t) / 60, 2)))
string_map5 = {

    'IFN-Œ±2b' : 'IFN-a2b',

    'TGF-Œ≤' : 'TGF-beta',

    'TNFŒ±' : 'TNFalpha',

    'Masson‚Äôs' : "Masson's",

    '3‚Äì7' : '3-7',

    '3‚Äì14' : '3-14',

    '6‚Äì8' : '6-8',

    '7‚Äì9' : '7-9',

    'product‚Äô' : "product'",

    'day‚Ä≤s' : " day's",

    'its‚Ä≤' : "its'",

    'Ôºö' : ':',

    '7‚Äì10' : '7-10',

    'Œºg' : 'µg',

    '‚Äì ' : '-',

    'patients‚Äô' : "patients'"

}



def generate_sentences(text):

    doc=sent_tokenize(text)

    sentences = []

    for i, token in enumerate(doc):

        sentence=scispacy_tokenizer(token)

        sentence=' '.join(sentence)

        sentences.append(sentence)

    return sentences



def build_similarity_matrix(sentences):

    # Create an empty similarity matrix

    similarity_matrix = np.zeros((len(sentences), len(sentences)))

 

    for idx1 in range(len(sentences)):

        for idx2 in range(len(sentences)):

            if idx1 == idx2: #ignore if both are same sentences

                continue 

            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])

    return similarity_matrix



def readability(text):

    for key, value in string_map5.items():

        text = text.replace(key, value)

    return text



def sentence_similarity(text1, text2):

    base = nlp(text1)

    compare = nlp(text2)

    return base.similarity(compare)



def generate_summary(file_name):

    summarize_text = []

    i=0

    

    # Step 1 - Read text, tokenize and split into sentenance

    sentences=generate_sentences(file_name)

    

    # Step 2 - Generate Similary Martix across sentences

    sentence_similarity_martix = build_similarity_matrix(sentences)

    

    # Step 3 - Rank sentences in similarity martix

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)

    scores = nx.pagerank(sentence_similarity_graph)

    

    k = Counter(scores)

    high = k.most_common(5) 

    high = sorted(high, reverse=False)

    

    # Step 4 - Sort the rank and pick top original sentences

    original_sentences=readability(file_name)

    original_sentences=sent_tokenize(original_sentences)

    

    # Step 5 - Pick top original sentences  

    sents=len(original_sentences)

    if sents <= 5:

        summarize_text=original_sentences

    elif sents > 5:

        top_n = 5

        for i in range(top_n):

            summarize_text.append("".join(original_sentences[high[i][0]]))

            

        

    # Step 6 - Return summarized text

    return " ".join(summarize_text)
def process_query(query): # Selecting relevant literature

    

    def get_top_tf_idf_words(tf_idf_vector, top_n=2):

        sorted_nzs = np.argsort(tf_idf_vector.data)[:-(top_n+1):-1]

        return feature_names[tf_idf_vector.indices[sorted_nzs]]

    

    disease2 =['covid',

               'covid-19',

               '-cov2', 

               'cov2',

               'sars-cov-2',

               '2019-ncov']

    

    pattern2 = '|'.join(disease2)

    

    # Calculate tfidf

    response_vector=cv.transform([query])

    tdtfi_response_vector=tfidf_transformer.transform(response_vector)

    

    # get top 15 tfidf words

    mylist = [get_top_tf_idf_words(tf_idf_vector,15) for tf_idf_vector in tdtfi_response_vector]

    feature_names2 = np.array(mylist)

    feature_names2 = np.squeeze(feature_names2)

    

    # keep only documents with relation to the query

    corpus_index = df.index

    matrix = pd.DataFrame(tf_idf_vector.todense(), index=corpus_index, columns=feature_names)

    test = pd.DataFrame(matrix, columns=feature_names2)

    

    test['QueryTotal']= test.sum(axis=1)

    test = test.loc[test['QueryTotal'] > 0.4]

    test = test.sort_values(by=['QueryTotal'],ascending=False)

    relevant_corpus_index = test.index

    

  

    # keep only documents similar to the query

    if len(relevant_corpus_index) > 0:

        df_similar = df.loc[relevant_corpus_index]

        df_similar['abstract_tokenised'] = df_similar.apply(

            lambda row: scispacy_tokenizer(row['abstract_processed']),axis=1)

        df_similar['abstract_tokenised'] = df_similar.apply(

            lambda row:' '.join(row['abstract_tokenised']),axis=1)

        df_similar['query_tokenised'] = query

        df_similar['query_tokenised'] = df_similar.apply(

            lambda row: scispacy_tokenizer(row['query_tokenised']),axis=1)

        df_similar['query_tokenised'] = df_similar.apply(

            lambda row:' '.join(row['query_tokenised']),axis=1)

        df_similar['similarity'] = df_similar.apply(

            lambda row: sentence_similarity(row['abstract_tokenised'],row['query_tokenised']),axis=1)

        

        # keep only documents with relation to the COVID-19

        df_similar = df_similar.loc[df['abstract_processed'].str.contains(pattern2)]

        df_similar = df_similar.sort_values(by=['similarity'],ascending=False)

        

        if len(df_similar) > 0:

            

            # Pick top 15 similar abstracts to query and perform extractive summarisation

            df_similar_condes = df_similar.loc[:,['title',

                                                  'authors',

                                                  'journal',

                                                  'abstract',

                                                  'url']].copy()

            pd.set_option('display.max_colwidth', 0)

            df_extract_sum=df_similar_condes[:15].copy()

            df_extract_sum['extractive_summary_of_abstract']=df_extract_sum.apply(

                lambda row: generate_summary(row['abstract']),axis=1)

            df_extract_sum1 = df_extract_sum.loc[:,['title',

                                                    'authors',

                                                    'journal',

                                                    'extractive_summary_of_abstract',

                                                    'url']].copy()

            return df_extract_sum1

        

        elif len(df_similar) <= 0:

            

            return 'No relevant artcles found.'

    

    elif len(relevant_corpus_index) <= 0:

        

        return 'No relevant artcles found.'
queries1 = ['Remdesivir effective against COVID-19?',

            'Favipiravir effective against COVID-19?',

            'Ribavirin effective against COVID-19?',

            'Lopinavir-ritonavir effective against COVID-19?',

            'Arbidol effective against COVID-19?']
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)



t = time()

length = len(queries1)

i=0



for i in range(length):

    pd.set_option('display.max_rows', None)

    df_summary=process_query(queries1[i])  

    if type(df_summary) == str:

        summary_html=df_summary

    elif type(df_summary) != str:

        summary_html=HTML(df_summary.to_html(escape=False,index=False))

    display(HTML('<h3>'+'Query: '+queries1[i]+'</h3>'), summary_html)

    

print('Time to preprocess everything: {} mins'.format(round((time() - t) / 60, 2)))
queries2 = ['Effectiveness of drugs being developed and tried to treat COVID-19 patients?',

            'Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocycline that may exert effects on viral replication?',

            'Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients?',

            'Exploration of use of best animal models and their predictive value for a human vaccine?',

            'Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents?',

            'Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need?',

            'Efforts targeted at a universal coronavirus vaccine?',

            'Efforts to develop animal models and standardize challenge studies?',

            'Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers?',

            'Approaches to evaluate risk for enhanced disease after vaccination?',

            'Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models in conjunction with therapeutics?']
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)



t = time()

length = len(queries2)

i=0



for i in range(length):

    pd.set_option('display.max_rows', None)

    df_summary=process_query(queries2[i])  

    if type(df_summary) == str:

        summary_html=df_summary

    elif type(df_summary) != str:

        summary_html=HTML(df_summary.to_html(escape=False,index=False))

    display(HTML('<h3>'+queries2[i]+'</h3>'), summary_html)

    

print('Time to preprocess everything: {} mins'.format(round((time() - t) / 60, 2)))