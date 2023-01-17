from nltk.corpus import stopwords                         # to load stopwords and remove the stopwords from text

from nltk import wordpunct_tokenize                       # Tokenize text 

from nltk.tokenize import RegexpTokenizer                 # Tokenize text by regular expression

from nltk.stem.porter import PorterStemmer                # convert words to root words like cats to cat

from nltk.stem.wordnet import WordNetLemmatizer           # the same portertstemmer

# FOR CLASSIFIAR

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.naive_bayes import GaussianNB



from torch.autograd import Variable             # for transfer inputs to autograde vaiable to requires grad

from string import punctuation                  # load punctuation like (">?/][]!`") and remove then from text

from gensim.models import Word2Vec             # to load word vectors to represnet any text to numerical numbers 



import pandas as pd                            # to read file as dataframe

import numpy as np                             # Linear algebra library

import torch                                   # pytorch Framework 

import torch.nn as nn                          # import neural netowrk from torch framework

import collections

import nltk

import torch.nn.functional as F

import matplotlib.pyplot as plt

import re

import string
def clean(text):

    text=text.lower()

    stp=set(stopwords.words("english"))

    placesp = re.compile('[/(){}\[\]\|@,;]')

    removech= re.compile('[^0-9a-z #+_]')

    st=WordNetLemmatizer()

    text=re.sub(placesp,' ',text)

    text=re.sub(removech,' ',text)

    text=text.split()

    text=[w for w in text if not w in stp]

    text=[st.lemmatize(w) for w in text]

    text=" ".join(text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    return text
"""

 Read Glove File take url of file return the two dictionaries ( word to index and word to vector in embedding )

 and one list of index to word  

 (glove file url) --> words_to_index, index_to_words, word_to_vec_map

 

 """

def read_glove_vecs(glove_file):

    with open(glove_file, 'r',encoding='UTF-8') as f:

        words = set()

        word_to_vec_map = {}

        for line in f:

            line = line.strip().split()

            curr_word = line[0]

            words.add(curr_word)

            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        

        i = 1

        words_to_index = {}

        index_to_words = {}

        for w in sorted(words):

            words_to_index[w] = i

            index_to_words[i] = w

            i = i + 1

    return words_to_index, index_to_words, word_to_vec_map
"""

Word Embeddings of words take dictionary of word to embedding and word to index

and return Embeddings Matrix [index,Embedding] 



"""



def pretrained_embedding_layer(word_to_vec_map, word_to_index):

    vocab_len = len(word_to_index) + 1

    emb_matrix = np.zeros((vocab_len,300))

    for word, index in word_to_index.items():

        emb_matrix[index, :] = word_to_vec_map[word]

    return emb_matrix
"""

Transfer sentence to indeces word in Embedding

take text and word to index dictionary 

return list of indeces word in Embedding



"""

def transfer_sent(text,word_to_index):

    text=text.split(' ')

    ret=[]

    for w in text:

        if w in word_to_index and w !="":

            ret.append(word_to_index[w])

    return ret
"""

calculate the Max Length in every column in Data Frame 

take Data Frame 

return Max lenght of columns



"""



def retmax(dftrain):



    lomax,temax,kemax=0,0,0

    for i in range(dftrain.shape[0]):



        temax=max(temax,len(np.array(dftrain.loc[i,'text'])))



        kemax=max(kemax,len(np.array(dftrain.loc[i,'keyword'])))



        lomax=max(lomax,len(np.array(dftrain.loc[i,'location'])))



        return kemax,lomax,temax
"""

Convert Data Frame to Matrix 2D by Adding padding zeros to every columns that not have lenght not equal max

lenght.

take Data Frame list of Max Lenghts of Columns

return Matrix after convert



"""



def convert2D(Xs,max_lens):

    

    X_indices = np.zeros((Xs[0].shape[0], sum(max_lens)))

    pls=0

    for i in range(Xs[0].shape[0]):

        pls=0

     

        for j in range(0,len(Xs[0][i])):

            X_indices[i][j+pls]=Xs[0][i][j]

        pls=max_lens[0]



        for j in range(0,len(Xs[1][i])):

            X_indices[i][j+pls]=Xs[1][i][j]

        pls=max_lens[1]+max_lens[0]



        for j in range(0,len(Xs[2][i])):

            X_indices[i][j+pls]=Xs[2][i][j]

    return X_indices
!ls "../input/glove6b300dtxt"
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

dftest = pd.read_csv("../input/nlp-getting-started/test.csv")

dftrain = pd.read_csv("../input/nlp-getting-started/train.csv")
print("shape of data",dftrain.shape)

dftrain.head(5)
print("Number of NAN value in keyword",dftrain.keyword.isnull().sum())

print("Number of NAN value in location",dftrain.location.isnull().sum())
dftrain.head()

dftrain.keyword.value_counts()
dftrain=dftrain.drop(['id'],axis=1)
"""

Call read_glove_vecs function and then call pretrained_embedding_layer to calc word Embedding of Words



"""



word_to_index, index_to_word, word_to_vec_map = read_glove_vecs("../input/glove6b300dtxt/glove.6B.300d.txt")

word_embedding=pretrained_embedding_layer(word_to_vec_map, word_to_index)
dftrain=dftrain.dropna(axis=0)

labels=dftrain.target

dftrain=dftrain.reset_index()
kews=dftest.keyword.values

kews=list(set(kews))

locs=dftest.location.values

locs=list(set(locs))
# TO CLEAN data using Function clean 

for i in range(dftrain.shape[0]):

    dftrain.at[i,'keyword']=transfer_sent(clean(dftrain.loc[i,'keyword']),word_to_index)

    dftrain.at[i,'location']=transfer_sent(clean(dftrain.loc[i,'location']),word_to_index)

    dftrain.at[i,'text']=transfer_sent(clean(dftrain.loc[i,'text']),word_to_index)

    

for i in range(dftest.shape[0]):

    if type(dftest.loc[0,'keyword'])==float:

        dftest.at[0,'keyword']=kews[np.random.randint(1,221)]

    if type(dftest.loc[0,'location'])==float:

        dftest.at[0,'location']=locs[np.random.randint(1,1602)]

        

    

    dftest.at[i,'keyword']=transfer_sent(clean(str(dftest.loc[i,'keyword'])),word_to_index)

    dftest.at[i,'location']=transfer_sent(clean(str(dftest.loc[i,'location'])),word_to_index)

    dftest.at[i,'text']=transfer_sent(clean(dftest.loc[i,'text']),word_to_index)

    

    
kemax,lomax,temax=retmax(dftrain)
print("Max len of text",temax)

print("Max len of location",lomax)

print("Max len of keyword",kemax)



dftrain=np.array(convert2D([dftrain.keyword,dftrain.location,dftrain.text],[0,0,25]),dtype=np.int64)

dftest=np.array(convert2D([dftest.keyword,dftest.location,dftest.text],[0,0,25]),dtype=np.int64)
model = OneVsRestClassifier(LogisticRegression(penalty='l2', C=1.0))
model.fit(dftrain,labels)
y_pred=model.predict(dftest)
sample_submission['target']=y_pred
sample_submission.to_csv("submission.csv", index=False)