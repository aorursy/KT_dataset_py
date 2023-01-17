# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

!pip install pandas==1.0.3
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
import scispacy
import en_core_sci_lg

import nltk
from nltk.corpus import stopwords

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
nlp = en_core_sci_lg.load()
stop_words = stopwords.words('english')
treat_synonyms = [
    'treat',
    'treatment', 
    'alleviate', 
    'manage', 
    'suppress',
    'suppression', 
    'prescribe',
    'therapy',
    'cure',
    'remedy', 
    'candidate', 
    'therapeutic',
    'administer',
    'deliver',
    'inject',
    'take',
    'effect',
    'inhale',
    'inhalation',
    'administration',
    'rescue',
    'reduce',
    'improve']
treat_vectors = {treat_synonym: nlp(treat_synonym).vector for treat_synonym in treat_synonyms}
directory = '../input/coronawhy/v6_text/v6_text/'
filenames = os.listdir(directory)
words_vectors = {}

for filename in filenames:
    filepath = os.path.join(directory, filename)
    print(filepath)
    df = pd.read_pickle(filepath, 'gzip')
    df['sentence'] = df['sentence'].str.lower()
    df['sentence'] = df['sentence'].str.replace('[{}]'.format(string.punctuation), '')
    df['sentence'] = df['sentence'].str.replace('|'.join(stop_words), '')
    
    for i, sentence in enumerate(df['sentence']):
        if not isinstance(sentence, str):
            continue
        for word in sentence.split():
            if word not in nlp.vocab:
                continue
            if word in words_vectors:
                continue

            vector = nlp(word).vector
            words_vectors[word] = vector
words_vectors = pd.DataFrame(words_vectors).T
words_vectors = words_vectors[~(words_vectors == 0).all(axis=1)]
words_vectors.to_csv('treawords_vectors.csv')
from sklearn.metrics.pairwise import cosine_similarity
df_similar_words = {}
for treat_word, vector in treat_vectors.items():
    word_cosine_similarity = cosine_similarity(X=words_vectors, Y=np.atleast_2d(vector))
    similar_words = pd.DataFrame(word_cosine_similarity, index=words_vectors.index).sort_values(by=0, ascending=False).head(10)[0]
    df_similar_words[treat_word] = similar_words
df_similar_words
unique_similar_words = {
    'treat': ['curb', 'expel', 'help', 'prune'],
    'treatment': [],
    'alleviate': [],
    'manage': ['expunge'],
    'suppress': ['engulf'],
    'suppression': [],
    'prescript': ['refer'],
    'therapy': [],
    'cure': ['fluence'],
    'remedy': [],
    'candidate': [],
    'therapeutic': ['new', 'newer'],  # treatments talked about in research papers may be described as "new" or "newer",
    'administer': [],
    'deliver': [],
    'inject': [],
    'take': [],
    'effect': [],
    'inhale': ['huff', 'puff', 'gulp'],
    'inhalation': ['burner'],
    'administration': [],
    'rescue': [],
    'reduce': ['fewer', 'quench'],
    'improve': []
}
