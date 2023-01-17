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
# dfにデータを取り込み

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

import time

sTime = time.time()

df_train=pd.read_csv( '../input/Competition_Train_Data_w_translation.txt', delimiter='\t' )
df_test=pd.read_csv('../input/Competition_Test_Data_w_translation.txt', delimiter='\t' )
df=pd.concat([df_train,df_test],sort=False,ignore_index=True)

#df=pd.concat([df[:5],df[-5:]],sort=False)

#word_vectors = KeyedVectors.load_word2vec_format('../input/googlenews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)

#print("df.loc[1][3]=",df.loc[1][3])
#doc = nlp(df.loc[1][3])
#print('doc=',doc)
#for token in doc:
#    print([token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#          token.shape_, token.is_alpha, token.is_stop, token.pos])


print("Finished -", time.time() - sTime)
# nlp

sTime = time.time()
nlp = spacy.load('en_core_web_sm')

df['nlp_doc_1'] = pd.Series(df['#1 String']).apply(lambda x: nlp(x))
df['nlp_doc_2'] = pd.Series(df['#2 String']).apply(lambda x: nlp(x))
print(time.time() - sTime, "sec")
df['nlp_doc_jpen_1'] = pd.Series(df['JPEN_1']).apply(lambda x: nlp(x))
df['nlp_doc_jpen_2'] = pd.Series(df['JPEN_2']).apply(lambda x: nlp(x))
print(time.time() - sTime, "sec")
df['nlp_doc_eoen_1'] = pd.Series(df['EOEN_1']).apply(lambda x: nlp(x))
df['nlp_doc_eoen_2'] = pd.Series(df['EOEN_2']).apply(lambda x: nlp(x))

#print(time.time() - sTime, "sec")


print("Finished -", time.time() - sTime)
df


# 特徴量の追加

sTime = time.time()

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
    # vector=doc[0].vector*0 # ← docはnlp_docの間違い？
    vector=nlp_doc[0].vector*0
    for token in nlp_doc:
        if token.pos_ == 'VERB':
            vector=vector + token.vector
            n=n+1
    return vector/n

def GetPropnMeanVector(nlp_doc):
    #固有名詞だけ抜き出した平均ベクトルを返す
    n=0
    vector=nlp_doc[0].vector*0
    for token in nlp_doc:
        if token.pos_ == 'PROPN':
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
        if len(t) >= 2:
            return 1
        return 0
    l = []
    for ss in s.split():
        if chkNumeric(ss):
            l.append(trimNum(ss))
    return l

def difList(a, b):
    # 共通部分を除いたときの左右それぞれの残存数
    # ただしどちらも空の場合は-1,-1を返す
    if len(a) + len(b) == 0:
        return -1, -1
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

df=df.assign(Similarity = df.apply(lambda x: x['nlp_doc_1'].similarity(x['nlp_doc_2']),axis=1))
df=df.assign(Similarity_jpen = df.apply(lambda x: x['nlp_doc_jpen_1'].similarity(x['nlp_doc_jpen_2']),axis=1))
df=df.assign(Similarity_jpen_diff = df.apply(lambda x: x['Similarity_jpen']-x['Similarity'],axis=1))
df=df.assign(Similarity_eoen = df.apply(lambda x: x['nlp_doc_eoen_1'].similarity(x['nlp_doc_eoen_2']),axis=1))
print(time.time() - sTime, "sec")

#Tokenize
df['Split_1'] = list(pd.Series(df['#1 String']).apply(lambda x: x.split(" ")))
df['Split_2'] = list(pd.Series(df['#2 String']).apply(lambda x: x.split(" ")))
df['no_of_word_1']=list(pd.Series(df['Split_1']).apply(lambda x: len(x)))
df['no_of_word_2']=list(pd.Series(df['Split_2']).apply(lambda x: len(x)))
df=df.assign(no_of_word_diff = df.apply(lambda x: (x['no_of_word_1']-x['no_of_word_2'])**2,axis=1))
print(time.time() - sTime, "sec")

# ----- Original -----
df['Nums_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: PickNums(x))
df['Nums_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: PickNums(x))
df['Propn_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: PickPropn(x))
df['Propn_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: PickPropn(x))
df['verbs_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: PickVerbs(x))
df['verbs_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: PickVerbs(x))
df['verb_mean_vector_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: GetVerbMeanVector(x))
df['verb_mean_vector_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: GetVerbMeanVector(x))
print(time.time() - sTime, "sec")
df=df.assign(VerbsSimilarity = df.apply(lambda x: GetSimilarity(x['verb_mean_vector_1'],(x['verb_mean_vector_2'])),axis=1))
df['VerbsSimilarity']=df['VerbsSimilarity'].fillna(0) #30個くらい何故かnaになるので0埋め

df['propn_mean_vector_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: GetPropnMeanVector(x))
df['propn_mean_vector_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: GetPropnMeanVector(x))
print(time.time() - sTime, "sec")
df=df.assign(PropnsSimilarity = df.apply(lambda x: GetSimilarity(x['propn_mean_vector_1'],(x['propn_mean_vector_2'])),axis=1))
df['PropnsSimilarity']=df['PropnsSimilarity'].fillna(0) #30個くらい何故かnaになるので0埋め

df=df.assign(PropnsDif = df.apply(lambda x: sum(difList(x['Propn_1'],(x['Propn_2']))),axis=1))


#df=df.assign(VerbsSimilarity = df.apply(lambda x: x['verbs_1'].similarity(x['verbs_2']),axis=1))
df['Lemma_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: ConvertToLemma(x))
df['Lemma_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: ConvertToLemma(x))
df['Pos_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: ConvertToPos(x))
df['Pos_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: ConvertToPos(x))

df['NumList_1'] = pd.Series(df['nlp_doc_1']).apply(lambda x: NumList(x))
df['NumList_2'] = pd.Series(df['nlp_doc_2']).apply(lambda x: NumList(x))
df=df.assign(NumListDif_1 = df.apply(lambda x: difList(x['NumList_1'],(x['NumList_2']))[0],axis=1))
df=df.assign(NumListDif_2 = df.apply(lambda x: difList(x['NumList_1'],(x['NumList_2']))[1],axis=1))
df=df.assign(NumListDif_diff = df.apply(lambda x: (x['NumListDif_1']-x['NumListDif_2'])**2,axis=1))


# ----- JPEN -----
df['Nums_jpen_1'] = pd.Series(df['nlp_doc_jpen_1']).apply(lambda x: PickNums(x))
df['Nums_jpen_2'] = pd.Series(df['nlp_doc_jpen_2']).apply(lambda x: PickNums(x))
df['Propn_jpen_1'] = pd.Series(df['nlp_doc_jpen_1']).apply(lambda x: PickPropn(x))
df['Propn_jpen_2'] = pd.Series(df['nlp_doc_jpen_2']).apply(lambda x: PickPropn(x))
df['verbs_jpen_1'] = pd.Series(df['nlp_doc_jpen_1']).apply(lambda x: PickVerbs(x))
df['verbs_jpen_2'] = pd.Series(df['nlp_doc_jpen_2']).apply(lambda x: PickVerbs(x))
df['verb_mean_vector_jpen_1'] = pd.Series(df['nlp_doc_jpen_1']).apply(lambda x: GetVerbMeanVector(x))
df['verb_mean_vector_jpen_2'] = pd.Series(df['nlp_doc_jpen_2']).apply(lambda x: GetVerbMeanVector(x))
print(time.time() - sTime, "sec")
df=df.assign(VerbsSimilarity_jpen = df.apply(lambda x: GetSimilarity(x['verb_mean_vector_jpen_1'],(x['verb_mean_vector_jpen_2'])),axis=1))
df['VerbsSimilarity_jpen']=df['VerbsSimilarity_jpen'].fillna(0) #30個くらい何故かnaになるので0埋め
df=df.assign(VerbsSimilarity_jpen_diff = df.apply(lambda x: x['VerbsSimilarity_jpen']-x['VerbsSimilarity'],axis=1))

df['propn_mean_vector_jpen_1'] = pd.Series(df['nlp_doc_jpen_1']).apply(lambda x: GetPropnMeanVector(x))
df['propn_mean_vector_jpen_2'] = pd.Series(df['nlp_doc_jpen_2']).apply(lambda x: GetPropnMeanVector(x))
print(time.time() - sTime, "sec")
df=df.assign(PropnsSimilarity_jpen = df.apply(lambda x: GetSimilarity(x['propn_mean_vector_jpen_1'],(x['propn_mean_vector_jpen_2'])),axis=1))
df['PropnsSimilarity_jpen']=df['PropnsSimilarity_jpen'].fillna(0) #30個くらい何故かnaになるので0埋め

#df=df.assign(VerbsSimilarity = df.apply(lambda x: x['verbs_1'].similarity(x['verbs_2']),axis=1))
df['Lemma_jpen_1'] = pd.Series(df['nlp_doc_jpen_1']).apply(lambda x: ConvertToLemma(x))
df['Lemma_jpen_2'] = pd.Series(df['nlp_doc_jpen_2']).apply(lambda x: ConvertToLemma(x))
df['Pos_jpen_1'] = pd.Series(df['nlp_doc_jpen_1']).apply(lambda x: ConvertToPos(x))
df['Pos_jpen_2'] = pd.Series(df['nlp_doc_jpen_2']).apply(lambda x: ConvertToPos(x))


# ----- EOEN -----
df['Nums_eoen_1'] = pd.Series(df['nlp_doc_eoen_1']).apply(lambda x: PickNums(x))
df['Nums_eoen_2'] = pd.Series(df['nlp_doc_eoen_2']).apply(lambda x: PickNums(x))
df['Propn_eoen_1'] = pd.Series(df['nlp_doc_eoen_1']).apply(lambda x: PickPropn(x))
df['Propn_eoen_2'] = pd.Series(df['nlp_doc_eoen_2']).apply(lambda x: PickPropn(x))
df['verbs_eoen_1'] = pd.Series(df['nlp_doc_eoen_1']).apply(lambda x: PickVerbs(x))
df['verbs_eoen_2'] = pd.Series(df['nlp_doc_eoen_2']).apply(lambda x: PickVerbs(x))
df['verb_mean_vector_eoen_1'] = pd.Series(df['nlp_doc_eoen_1']).apply(lambda x: GetVerbMeanVector(x))
df['verb_mean_vector_eoen_2'] = pd.Series(df['nlp_doc_eoen_2']).apply(lambda x: GetVerbMeanVector(x))
print(time.time() - sTime, "sec")
df=df.assign(VerbsSimilarity_eoen = df.apply(lambda x: GetSimilarity(x['verb_mean_vector_eoen_1'],(x['verb_mean_vector_eoen_2'])),axis=1))
df['VerbsSimilarity_eoen']=df['VerbsSimilarity_eoen'].fillna(0) #30個くらい何故かnaになるので0埋め
df=df.assign(VerbsSimilarity_eoen_diff = df.apply(lambda x: x['VerbsSimilarity_eoen']-x['VerbsSimilarity'],axis=1))

df['propn_mean_vector_eoen_1'] = pd.Series(df['nlp_doc_eoen_1']).apply(lambda x: GetPropnMeanVector(x))
df['propn_mean_vector_eoen_2'] = pd.Series(df['nlp_doc_eoen_2']).apply(lambda x: GetPropnMeanVector(x))
print(time.time() - sTime, "sec")
df=df.assign(PropnsSimilarity_eoen = df.apply(lambda x: GetSimilarity(x['propn_mean_vector_eoen_1'],(x['propn_mean_vector_eoen_2'])),axis=1))
df['PropnsSimilarity_eoen']=df['PropnsSimilarity_eoen'].fillna(0) #30個くらい何故かnaになるので0埋め

#df=df.assign(VerbsSimilarity = df.apply(lambda x: x['verbs_1'].similarity(x['verbs_2']),axis=1))
df['Lemma_eoen_1'] = pd.Series(df['nlp_doc_eoen_1']).apply(lambda x: ConvertToLemma(x))
df['Lemma_eoen_2'] = pd.Series(df['nlp_doc_eoen_2']).apply(lambda x: ConvertToLemma(x))
df['Pos_eoen_1'] = pd.Series(df['nlp_doc_eoen_1']).apply(lambda x: ConvertToPos(x))
df['Pos_eoen_2'] = pd.Series(df['nlp_doc_eoen_2']).apply(lambda x: ConvertToPos(x))


print(time.time() - sTime, "sec")

print("Finished -", time.time() - sTime)
df

import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
from textblob import TextBlob

sTime = time.time()

nlp = spacy.load('en', parse=True, tag=True, entity=True)
#nlp_vec = spacy.load('en_vecs', parse = True, tag=True, #entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

print(time.time() - sTime, "sec")

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def normalize_corpus(corpus, html_stripping=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

df['#1_Clean'] = normalize_corpus(df['#1 String'])
df['#2_Clean'] = normalize_corpus(df['#2 String'])
print(time.time() - sTime, "sec")

print("Finished -", time.time() - sTime)

sTime = time.time()

df=df.assign(Vsentiment_scores_1 = df.apply(lambda x: round(TextBlob(x['#1_Clean']).sentiment.polarity, 3),axis=1))
df=df.assign(Vsentiment_scores_2 = df.apply(lambda x: round(TextBlob(x['#2_Clean']).sentiment.polarity, 3),axis=1))
df=df.assign(Vsentiment_scores_diff = df.apply(lambda x: (x['Vsentiment_scores_1']-x['Vsentiment_scores_2'])**2,axis=1))
df=df.assign(Vsentiment_scores_diff_excl0 = df.apply(lambda x: 0 if x['Vsentiment_scores_1']==0 or x['Vsentiment_scores_2']==0 else (x['Vsentiment_scores_1']-x['Vsentiment_scores_2'])**2,axis=1))

print(time.time() - sTime, "sec")


df=df.assign(Similarity_max = df.apply(lambda x: max(x['Similarity'],(x['Similarity_jpen']),(x['Similarity_eoen'])),axis=1))
df=df.assign(VerbsSimilarity_max = df.apply(lambda x: max(x['VerbsSimilarity'],(x['VerbsSimilarity_jpen']),(x['VerbsSimilarity_eoen'])),axis=1))
df=df.assign(PropnsSimilarity_max = df.apply(lambda x: max(x['PropnsSimilarity'],(x['PropnsSimilarity_jpen']),(x['PropnsSimilarity_eoen'])),axis=1))
df=df.assign(Similarity_max_max = df.apply(lambda x: max(x['Similarity_max'],(x['VerbsSimilarity_max'])),axis=1))


print("Finished -", time.time() - sTime)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report

x_vars = ['no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity','Similarity_jpen_diff','VerbsSimilarity_jpen_diff','NumListDif_1','NumListDif_2']
x_vars = ['no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity','Similarity_jpen_diff','VerbsSimilarity_jpen_diff','NumListDif_1','NumListDif_2']
x_vars = ['no_of_word_1','no_of_word_2','Similarity_jpen','VerbsSimilarity_jpen']
x_vars = ['no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity','NumListDif_1','NumListDif_2']
x_vars = ['no_of_word_diff','Similarity','VerbsSimilarity','NumListDif_1','NumListDif_2'] # 73
x_vars = ['no_of_word_diff','Similarity','VerbsSimilarity','NumListDif_1','NumListDif_2','Vsentiment_scores_1','Vsentiment_scores_2'] # 72
x_vars = ['no_of_word_diff','VerbsSimilarity','NumListDif_1','NumListDif_2','Vsentiment_scores_1','Vsentiment_scores_2'] # 73 (submission)
x_vars = ['no_of_word_diff','Similarity','Similarity_jpen','Similarity_eoen','VerbsSimilarity','VerbsSimilarity_jpen','VerbsSimilarity_eoen','NumListDif_diff','Vsentiment_scores_diff','VerbsSimilarity_jpen_diff']
# x_vars = ['no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity']
# x_vars = ['no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity']

# 説明変数、目的変数
X = df[df['Label']>=0][x_vars]
y = df[df['Label']>=0]['Label']
X_submit = df[df['Label'].isna()][x_vars]
# 学習用、検証用データ作成
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = 0.3, random_state = 666)

# XGboostのライブラリをインポート
import xgboost as xgb
# モデルのインスタンス作成
clf = xgb.XGBClassifier()

# ハイパーパラメータの探索
clf_cv = GridSearchCV(clf, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
clf_cv.fit(X_train, y_train)
print (clf_cv.best_params_, clf_cv.best_score_)

# 改めて最適パラメータで学習
clf = xgb.XGBClassifier(**clf_cv.best_params_)
clf.fit(X_train, y_train)

# 学習モデルの評価
pred = clf.predict(X_test)
print("学習モデルの評価")
print( confusion_matrix(y_test, pred))
print( classification_report(y_test, pred))

# Submissionデータの作成
pred_submit = clf.predict(X_submit)
Submission = pd.concat([df[df['Label'].isna()]['Unique ID'].reset_index()
                        ,pd.Series(pred_submit.T.astype(np.int32))] ,axis=1)
Submission.columns=['index','Unique ID','Label']
#Submission = Submission[['Unique ID','Label']]
#print("Submission= ",Submission.info())
#print("ID=",df[df['Label'].isna()]['Unique ID'])
#print("Submission data")
#print(Submission)
Submission = Submission[['Unique ID','Label']]
Submission.to_csv("Submission1.csv")
Submission

# For Check

# output
# df.to_csv("df.csv")
# print("OK")

# output small
df_small = df[["Label","Unique ID",'NumListDif_1','NumListDif_2',"Vsentiment_scores_1","Vsentiment_scores_2","Similarity","Similarity_jpen","Similarity_eoen","VerbsSimilarity","VerbsSimilarity_jpen","VerbsSimilarity_eoen"]]
df_small.to_csv("df_small.csv")
print("Finished")

df[:50][["Label","Unique ID",'NumListDif_1','NumListDif_2',"Vsentiment_scores_1","Vsentiment_scores_2","Similarity","Similarity_jpen","Similarity_eoen","VerbsSimilarity","VerbsSimilarity_jpen","VerbsSimilarity_eoen"]]
# Neural Network by Keras
# From http://aidiary.hatenablog.com/entry/20161108/1478609028

from sklearn import datasets
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn import preprocessing

def build_multilayer_perceptron(input_size,output_size):
    """多層パーセプトロンモデルを構築"""
    model = Sequential()
    model.add(Dense(10, input_shape=(input_size, )))
    model.add(Activation('relu'))
    model.add(Dense(6))
    model.add(Activation('relu'))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))
    return model


#features = ['diff_word','no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity']
#features = ['no_of_word_1','no_of_word_2','Similarity','VerbsSimilarity']
features = ['NumListDif_1','NumListDif_2',"Vsentiment_scores_1","Vsentiment_scores_2","Similarity","Similarity_jpen","Similarity_eoen","VerbsSimilarity","VerbsSimilarity_jpen","VerbsSimilarity_eoen"]
features = ['NumListDif_diff',"Vsentiment_scores_1","Vsentiment_scores_2","Similarity_max","VerbsSimilarity_max","PropnsDif"]

X = df[df['Label']>=0][features]
Y = df[df['Label']>=0]['Label']
X_submit = df[df['Label'].isna()][features]

if __name__ == "__main__":

    # データの標準化
    X = preprocessing.scale(X)

    # ラベルをone-hot-encoding形式に変換
    # 0 => [1, 0]
    # 1 => [0, 1]
    Y = np_utils.to_categorical(Y)

    # 訓練データとテストデータに分割
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
    print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

    # モデル構築
    model = build_multilayer_perceptron(len(features),2)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # モデル訓練
    model.fit(train_X, train_Y, nb_epoch=50, batch_size=1, verbose=1)

    # モデル評価
    loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
    print("Accuracy = {:.3f}".format(accuracy))
