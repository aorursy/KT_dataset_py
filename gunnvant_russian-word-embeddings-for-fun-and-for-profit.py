import os
import pandas as pd
import numpy as np
import glob
import nltk
import gensim
train=pd.read_csv("../input/avito-demand-prediction/train.csv")
train.head()
from gensim.models import KeyedVectors
ru_model = KeyedVectors.load_word2vec_format('../input/fasttext-russian-2m/wiki.ru.vec')
print("The size of vocabulary for this corpus is {}".format(len(ru_model.vocab)))
# Pick a word 
find_similar_to = 'Автомобили'.lower()
ru_model.similar_by_word(find_similar_to)
import nltk
def tokenize(x):
    '''Input: One description'''
    tok=nltk.tokenize.toktok.ToktokTokenizer()
    return [t.lower() for t in tok.tokenize(x)]
def get_vector(x):
    '''Input: Single token''' #If the word is out of vocab, then return a 300 dim vector filled with zeros
    try:
        return ru_model.get_vector(x)
    except:
        return np.zeros(shape=300)
def vector_sum(x):
    '''Input:List of word vectors'''
    return np.sum(x,axis=0)
features=[]
for desc in train['description'].values:
    tokens=tokenize(desc)
    if len(tokens)!=0: ## If the description is missing then return a 300 dim vector filled with zeros
        word_vecs=[get_vector(w) for w in tokens]
        features.append(vector_sum(word_vecs))
    else:
        features.append(np.zeros(shape=300))                 
print("Features were extracted from {} rows".format(len(features)))
## Convert into numpy array
train_desc_features=np.array(features)
print("Shape of features extracted from 'Description' column is:")
print(train_desc_features.shape)
## Write out as .npy file to be used later for modelling
## np.save("train_desc_features.npy",train_desc_features)
## Due to kernel limitations, this step fails, I had trained a file locally and can be accessed from:
## https://s3.us-east-2.amazonaws.com/datafaculty/train_desc_features.npy
