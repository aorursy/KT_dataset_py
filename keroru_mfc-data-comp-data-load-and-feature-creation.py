import pandas as pd
#tes=pd.read_csv('../input/test-eo/Competition_Test_Data_EO.txt')
tes=pd.DataFrame(['a','b','c'],[1,2,4])
tes.columns=['val']
tes['#1 String']=tes['val']
tes['Output_EO_1']=tes['#1 String']
tes
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

df_train=pd.read_csv( '../input/mfc-data-compe-20181014/Competition_Train_Data (003).txt', delimiter='\t' )
df_test=pd.read_csv('../input/mfc-data-compe-test-data/Competition_Test_Data.txt', delimiter='\t' )
df=pd.concat([df_train,df_test],sort=False,ignore_index=True)

#df=pd.concat([df[:5],df[-5:]],sort=False)

#word_vectors = KeyedVectors.load_word2vec_format('../input/googlenews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
nlp = spacy.load('en_core_web_sm')

df['nlp_doc_1'] = pd.Series(df['#1 String']).apply(lambda x: nlp(x))
df['nlp_doc_2'] = pd.Series(df['#2 String']).apply(lambda x: nlp(x))
df=df.assign(Similarity = df.apply(lambda x: x['nlp_doc_1'].similarity(x['nlp_doc_2']),axis=1))

#Tokenize
df['Split_1'] = list(pd.Series(df['#1 String']).apply(lambda x: x.split(" ")))
df['Split_2'] = list(pd.Series(df['#2 String']).apply(lambda x: x.split(" ")))
df['no_of_word_1']=list(pd.Series(df['Split_1']).apply(lambda x: len(x)))
df['no_of_word_2']=list(pd.Series(df['Split_2']).apply(lambda x: len(x)))

df

#print("df.loc[1][3]=",df.loc[1][3])
#doc = nlp(df.loc[1][3])
#print('doc=',doc)
#for token in doc:
#    print([token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#          token.shape_, token.is_alpha, token.is_stop, token.pos])
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
    vector=doc[0].vector*0
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

print(PickVerbs(doc))

df['Nums_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: PickNums(x))
df['Nums_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: PickNums(x))
df['Propn_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: PickPropn(x))
df['Propn_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: PickPropn(x))
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

df['VerbsSimilarity']=df['VerbsSimilarity'].fillna(0) #30個くらい何故かnaになるので0埋め

plt.hist(df[df['Label']==0]['Similarity'],histtype="step",label="Similarity_0")
plt.hist(df[df['Label']==1]['Similarity'],histtype="step",label="Similarity_1")
plt.hist(df[df['Label']==0]['VerbsSimilarity'],histtype="step",label="VerbsSimilarity_0")
plt.hist(df[df['Label']==1]['VerbsSimilarity'],histtype="step",label="VerbsSimilarity_1")

df.info()
#df.mean()
#df[60:65]
#df[df['Label']==0]


from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    words1 = '#1 String'
    words2 = '#2 String'
    q1words = {}
    q2words = {}
    for word in str(row[words1]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row[words2]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def Propn_match_share(row):
    words1 = 'Propn_1'
    words2 = 'Propn_2'
    q1words = {}
    q2words = {}
    for word in str(row[words1]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row[words2]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R

def Lemma_match_share(row):
    words1 = 'Lemma_1'
    words2 = 'Lemma_2'
    q1words = {}
    q2words = {}
    for word in str(row[words1]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row[words2]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def Nums_match_share(row):
    words1 = 'Nums_1'
    words2 = 'Nums_2'
    q1words = {}
    q2words = {}
    for word in str(row[words1]).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row[words2]).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


plt.figure(figsize=(15, 5))
train_word_match = df.apply(Nums_match_share, axis=1, raw=True)
df=df.assign(Nums_match = train_word_match)
plt.hist(train_word_match[df['Label'] == 0], bins=20, normed=True, label='Label = 0')
plt.hist(train_word_match[df['Label'] == 1], bins=20, normed=True, alpha=0.7, label='Label = 1')
plt.legend()
plt.title('Label distribution over Nums_match_share', fontsize=15)
plt.xlabel('Nums_match_share', fontsize=15)


plt.figure(figsize=(15, 5))
train_word_match = df.apply(Propn_match_share, axis=1, raw=True)
df=df.assign(Propn_match = train_word_match)
plt.hist(train_word_match[df['Label'] == 0], bins=20, normed=True, label='Label = 0')
plt.hist(train_word_match[df['Label'] == 1], bins=20, normed=True, alpha=0.7, label='Label = 1')
plt.legend()
plt.title('Label distribution over Propn_match_share', fontsize=15)
plt.xlabel('Propn_match_share', fontsize=15)


plt.figure(figsize=(15, 5))
train_word_match = df.apply(Lemma_match_share, axis=1, raw=True)
df=df.assign(Lemma_match = train_word_match)
plt.hist(train_word_match[df['Label'] == 0], bins=20, normed=True, label='Label = 0')
plt.hist(train_word_match[df['Label'] == 1], bins=20, normed=True, alpha=0.7, label='Label = 1')
plt.legend()
plt.title('Label distribution over Lemma_match_share', fontsize=15)
plt.xlabel('Lemma_match_share', fontsize=15)

plt.figure(figsize=(15, 5))
train_word_match = df.apply(word_match_share, axis=1, raw=True)
df=df.assign(word_match = train_word_match)
plt.hist(train_word_match[df['Label'] == 0], bins=20, normed=True, label='Label = 0')
plt.hist(train_word_match[df['Label'] == 1], bins=20, normed=True, alpha=0.7, label='Label = 1')
plt.legend()
plt.title('Label distribution over word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)


plt.figure(figsize=(15, 5))
simil = df.apply(lambda x: x['nlp_doc_1'].similarity(x['nlp_doc_2']),axis=1)
plt.hist(simil[df['Label'] == 0], bins=20, normed=True, label='Label = 0')
plt.hist(simil[df['Label'] == 1], bins=20, normed=True, alpha=0.7, label='Label = 1')
plt.legend()
plt.title('Label distribution over similarity', fontsize=15)
plt.xlabel('similarity', fontsize=15)

df.info()
df.to_csv("result.csv")
#fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
#df[0:150].plot(x='ad', y='ba', kind='scatter', ax=axes.flatten()[0])
#df.plot(x='Similarity', y='Label', kind='scatter', ax=axes.flatten()[0])
#df.mean()
#df_sample=df.head(20)