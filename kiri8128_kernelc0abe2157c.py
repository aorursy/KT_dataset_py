# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(2**10)
import numpy as np
import os
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors
import spacy
from spacy.tokens import Doc

df=pd.read_csv( '../input/Competition_Train_Data.txt', delimiter='\t' )
#df=df.head(6)

#word_vectors = KeyedVectors.load_word2vec_format('../input/googlenews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
nlp = spacy.load('en_core_web_sm')
df['nlp_doc_1'] = pd.Series(df['#1 String']).apply(lambda x: nlp(x))
df['nlp_doc_2'] = pd.Series(df['#2 String']).apply(lambda x: nlp(x))
df=df.assign(Similarity = df.apply(lambda x: x['nlp_doc_1'].similarity(x['nlp_doc_2']),axis=1))

#print("df.loc[1][3]=",df.loc[1][3])
doc = nlp(df.loc[1][3])
print('doc=',doc)
for token in doc:
    print([token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop, token.pos])

#Tokenize
df['Split_1'] = list(pd.Series(df['#1 String']).apply(lambda x: x.split(" ")))
df['Split_2'] = list(pd.Series(df['#2 String']).apply(lambda x: x.split(" ")))
df['no_of_word_1']=list(pd.Series(df['Split_1']).apply(lambda x: len(x)))
df['no_of_word_2']=list(pd.Series(df['Split_2']).apply(lambda x: len(x)))


#fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
#df[0:150].plot(x='ad', y='ba', kind='scatter', ax=axes.flatten()[0])
#df.plot(x='Similarity', y='Label', kind='scatter', ax=axes.flatten()[0])
#df.mean()
#df_sample=df.head(20)

print("End")
# 関数定義 および 特徴量の追加

#df['nlp_doc_1'] = pd.Series(df['#1 String']).apply(lambda x: nlp(x))
#nlp(df['#1 String'])
doc = nlp(df.loc[1][3])

def PickVerbs(nlp_doc):
    #動詞のみ原型で取り出す。Docで返す。
    verbs=[]
    for token in nlp_doc:
        if token.pos_ == 'VERB':
            verbs = verbs + [token.lemma_]
    doc = Doc(nlp.vocab, verbs)
    return doc
    #return verbs

def PickPropn(nlp_doc):
    #固有名詞のみ原型で取り出す
    verbs=[]
    for token in nlp_doc:
        if token.pos_ == 'PROPN':
            verbs = verbs + [token.lemma_]
    return verbs

def PickNums(nlp_doc):
    #数字のみ原型で取り出す
    verbs=[]
    for token in nlp_doc:
        if token.pos_ == 'NUM':
            verbs = verbs + [token.lemma_]
    return verbs


def GetVerbMeanVector(nlp_doc):
    #動詞だけ抜き出した平均ベクトルを返す
    n=0
    vector=doc[0].vector
    for token in nlp_doc:
        if token.pos_ == 'VERB':
            vector=vector + token.vector
            n=n+1
    return vector/n

def GetSimilarity(arr1, arr2):
    #単語Vectorの内積を返す
    return np.dot(arr1,arr2)/np.sqrt(np.dot(arr1,arr1)*np.dot(arr2,arr2))


def ConvertToLemma(nlp_doc):
    #原型を返す
    ls=[]
    for token in nlp_doc:
        ls = ls + [token.lemma_]
    return ls

def ConvertToPos(nlp_doc):
    #品詞を返す
    ls=[]
    for token in nlp_doc:
        ls = ls + [token.pos_]
    return ls

def ConvertToTag(nlp_doc):
    #タグを返す
    ls=[]
    for token in nlp_doc:
        ls = ls + [token.tag_]
    return ls

def ConvertToDep(nlp_doc):
    #Depを返す
    ls=[]
    for token in nlp_doc:
        ls = ls + [token.dep_]
    return ls

def NumList(s):
    s = str(s)
    def trimNum(s):
        ret = ""
        for ss in s:
            if ss in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                ret += ss
        return ret
    def chkNumeric(s):
        t = trimNum(s)
        if len(t) >= 1:
            return 1
        return 0
    l = []
    for ss in s.split():
        if chkNumeric(ss):
            l.append(trimNum(ss))
    return l

def difList(a, b):
    # 共通部分を除いたときの左右それぞれの残存数
    for aa in a:
        if aa in b:
            a.remove(aa)
            b.remove(aa)
            return difList(a, b)
    for bb in b:
        if bb in a:
            a.remove(bb)
            b.remove(bb)
            return difList(a, b)
    return len(a), len(b)

print(PickVerbs(doc))

df['Nums_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: PickNums(x))
df['Nums_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: PickNums(x))
df['verbs_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: PickVerbs(x))
df['verbs_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: PickVerbs(x))
df['verb_mean_vector_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: GetVerbMeanVector(x))
df['verb_mean_vector_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: GetVerbMeanVector(x))
df=df.assign(VerbsSimilarity = df.apply(lambda x: GetSimilarity(x['verb_mean_vector_1'],(x['verb_mean_vector_2'])),axis=1))
#df=df.assign(VerbsSimilarity = df.apply(lambda x: x['verbs_1'].similarity(x['verbs_2']),axis=1))
df['Lemma_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: ConvertToLemma(x))
df['Lemma_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: ConvertToLemma(x))
df['Pos_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: ConvertToPos(x))
df['Pos_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: ConvertToPos(x))
df['NumList_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: NumList(x))
df['NumList_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: NumList(x))
df=df.assign(NumListDif_1 = df.apply(lambda x: difList(x['NumList_1'],(x['NumList_2']))[0],axis=1))
df=df.assign(NumListDif_2 = df.apply(lambda x: difList(x['NumList_1'],(x['NumList_2']))[1],axis=1))


df['VerbsSimilarity']=df['VerbsSimilarity'].fillna(0) #30個くらい何故かnaになるので0埋め

plt.hist(df[df['Label']==0]['Similarity'],histtype="step",label="Similarity_0")
plt.hist(df[df['Label']==1]['Similarity'],histtype="step",label="Similarity_1")
plt.hist(df[df['Label']==0]['VerbsSimilarity'],histtype="step",label="VerbsSimilarity_0")
plt.hist(df[df['Label']==1]['VerbsSimilarity'],histtype="step",label="VerbsSimilarity_1")

df[:50]
# df[:3][["Label","verbs_1", "verbs_2"]]

# df.info()
#df.mean()
# df
#df[df['Label']==0]

#df.to_csv()
