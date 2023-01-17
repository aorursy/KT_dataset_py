import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/labeledTrainData.tsv',sep='\t',names=['id','rating','comments'])
df1=pd.read_csv('../input/labeledTrainData.tsv',sep='\t')
df.head()
df.groupby('rating').describe()
df1.head()
df['comments'][1]
df.describe()
df1.describe() # since sentiment is only number in df1
import nltk
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
ps=PorterStemmer()
string.punctuation
stopwords.words('english')[:10]
#returning a single paragraph containing filtered words
def filter_punc_stopwrds(txt):
    """
    Remove punctuation
    Remove stopwords
    """
    txtin=BeautifulSoup(txt,'lxml').get_text()
    nopunc=''.join( c for c in ps.stem(txtin.lower()) if c not in string.punctuation )
    return ' '.join( word for word in nopunc.split() if word not in stopwords.words('english'))
    #return ' '.join( lem.lemmatize(word) for word in nostop.split())
df['comments'][1:3].apply(filter_punc_stopwrds)
df1['review'][1:3].apply(filter_punc_stopwrds)
#returning filtered list of words in a review to be used in CountVectorizer
def filter_punc_stopwrds(txt):
    """
    Remove punctuation
    Remove stopwords
    """
    txtin=BeautifulSoup(txt,'lxml').get_text()
    nopunc=''.join( c for c in ps.stem(txtin.lower()) if c not in string.punctuation )
    return [ word for word in nopunc.split() if word not in stopwords.words('english')]
from sklearn.feature_extraction.text import CountVectorizer
transform_com=CountVectorizer(analyzer=filter_punc_stopwrds).fit(df['comments'])
transform_com1=CountVectorizer(analyzer=filter_punc_stopwrds).fit(df1['review'])
print(len(transform_com.vocabulary_))
transform_com.transform([df['comments'].min()])
len(df['comments'].min())
transform_com.get_feature_names()[138635]
transform_com.get_feature_names()[13401]
trans_com=transform_com.transform(df['comments'])
trans_com1=transform_com1.transform(df1['review'])
trans_com.shape
trans_com.nnz
sparsity=(100.0 * trans_com.nnz / (trans_com.shape[0] *  trans_com.shape[1] ))
print(sparsity)
sparsity=(100.0 * trans_com1.nnz / (trans_com1.shape[0] *  trans_com1.shape[1] ))
print(sparsity)
from sklearn.feature_extraction.text import TfidfTransformer
tdif_trans=TfidfTransformer().fit(trans_com)
tdif_trans1=TfidfTransformer().fit(trans_com1)
tdif1=tdif_trans.transform(trans_com)
tdif2=tdif_trans1.transform(trans_com1)
print(tdif_trans.idf_[transform_com.vocabulary_['g']])
print(tdif_trans.idf_[transform_com.vocabulary_['good']])
tdif1.shape
print(tdif_trans1.idf_[transform_com.vocabulary_['g']])
print(tdif_trans1.idf_[transform_com.vocabulary_['good']])
print(tdif1[0])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
X_train,X_test,Y_train,Y_test=train_test_split(tdif1,df['rating'],test_size=0.33,random_state=42)
clf=RandomForestClassifier(n_estimators=250)
clf=clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)# mean accuracy
X_train1,X_test1,Y_train1,Y_test1=train_test_split(tdif2,df1['sentiment'],test_size=0.33,random_state=42)
clf1=RandomForestClassifier(n_estimators=250)
clf1=clf1.fit(X_train1,Y_train1)
clf1.score(X_test1,Y_test1)# mean accuracy. the score is not much difference we can comapre here.
