import pandas as pd

import numpy as np

import spacy

import re

from time import time
df = pd.read_csv('../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')

df.head()
df = df[['Title', 'Review Text']] #I only want to use the text based values, so I modify the dataframe

df.head()
df.isnull().sum() #finding the null values
df = df.dropna().reset_index(drop=True) #and dropping them 

df.isnull().sum() 
df.shape
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
def cleaning(doc): #the lemmatizing function

    txt = [token.lemma_ for token in doc if not token.is_stop]

    if len(txt) > 2:

        return ' '.join(txt)
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['Review Text'])
t = time()



txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]



print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
df_clean = pd.DataFrame({'clean': txt})

df_clean = df_clean.dropna().drop_duplicates()

df_clean.shape
df_clean.head() 
import gensim 

from gensim.models import Word2Vec
sent = [row.split() for row in df_clean['clean']] #splitting the columns into the correct format
print(sent[:10])
t = time()



model = Word2Vec(sent, min_count=1,size= 50,workers=3, window =3, sg = 1)



print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
model.wv.most_similar(positive=["dress"])
model.wv.most_similar(positive=["jumper"])
model.wv.most_similar(positive=["skirt"])
model.wv.most_similar(positive=["favorite"])
model.wv.most_similar(positive=["favourite"])
model.wv.similarity("little", 'petite')
model.wv.similarity("pencil", 'skirt')
model.wv.doesnt_match(['skirt', 'dress', 'book'])