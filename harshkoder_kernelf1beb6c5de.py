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
import glob

import re

import os

import nltk

from nltk.stem.snowball import SnowballStemmer

from gensim import corpora

from nltk.corpus import stopwords

from gensim.models.tfidfmodel import TfidfModel

from gensim import similarities

import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
texts=[re.sub('[\W_]+',' ',open(f,encoding='utf-8-sig').read()) for f in glob.glob('../input/*.txt')]
stemmer = SnowballStemmer("english")

def tokem(text):

    

    # Tokenize by word

    tokens = [word.lower() for word in nltk.word_tokenize(text) if word not in stopwords.words('english')]

    

    # Stemming

    stems = [stemmer.stem(word) for word in tokens]

    

    return stems

corpus=[tokem(text) for text in texts]

dictionary = corpora.Dictionary(corpus)

bows = [dictionary.doc2bow(doc) for doc in corpus]

model = TfidfModel(bows)
sim_matrix=similarities.MatrixSimilarity(model[bows])

sim_df = pd.DataFrame(list(sim_matrix))
mtd=pd.read_csv('../input/metadata.tsv',sep='\t')

dic=dict(zip(mtd['Path'],mtd['Title']))

titles=[dic[os.path.basename(f)] for f in  glob.glob('../input/*.txt')]
sim_df.columns = titles

sim_df.index = titles

sim_df.head()
v=sim_df['The Life and Letters of Charles Darwin, Volume I (of II) Edited by His Son']

v_sorted = v.sort_values()

v_sorted.plot.barh(x='lab', y='val', rot=0).plot()
v=sim_df['Coral Reefs']

v_sorted = v.sort_values()

v_sorted.plot.barh(x='lab', y='val', rot=0).plot()
Z = hierarchy.linkage(sim_matrix, 'ward')

p=hierarchy.dendrogram(Z, leaf_font_size=9, labels=sim_df.index, orientation='left')

plt.show()