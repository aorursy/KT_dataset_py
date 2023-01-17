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
import numpy as np
import pandas as pd
import codecs
input_file = codecs.open('../input/socialmedia-disaster-tweets-DFE.csv', 'r',encoding='utf-8', errors='replace')
output_file = open('clean_socialmedia-disaster.csv', 'w')

def sanitize_characters(raw, clean):    
    for line in input_file:
        out = line
        output_file.write(line)
sanitize_characters(input_file, output_file)
df = pd.read_csv('clean_socialmedia-disaster.csv')
df.head()
df = df[df.choose_one != "Can't Decide"]
df = df[['text','choose_one']]
df['relevant'] = df.choose_one.map({'Relevant':1,'Not Relevant':0})
import spacy
nlp = spacy.load('en',disable=['tagger','parser','ner'])
from tqdm import tqdm, tqdm_notebook
tqdm.pandas(tqdm_notebook)

df['lemmas'] = df["text"].progress_apply(lambda row: 
                                         [w.lemma_ for w in nlp(row)])
df['joint_lemmas'] = df['lemmas'].progress_apply(lambda row: ' '.join(row))
df.head()
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=2,verbose=1)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
tf = vectorizer.fit_transform(df['joint_lemmas'])
lda.fit(tf)
tf_feature_names = vectorizer.get_feature_names()
n_top_words = 5
for topic_idx, topic in enumerate(lda.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([tf_feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
