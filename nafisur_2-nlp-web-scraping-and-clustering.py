import pandas as pd
import nltk
dataset=pd.read_csv('../input/allposts.csv',sep='\t',quoting=3)
dataset.head()
df=dataset[['post']]
df.head()
len(df)
df.info()
df=df.dropna()
len(df)
df[df['post']=='"'].head()
l=df[df['post']=='"'].index
df=df.drop(labels=l)
len(df)
df = df.reset_index(drop=True)
len(df)
df1=df
df.head()
from nltk.corpus import stopwords
from string import punctuation
import re
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
ls=WordNetLemmatizer()
cstopwords=set(stopwords.words('english')+list(punctuation))
text_corpus=[]
for i in range(0,len(df)):
    review=re.sub('[^a-zA-Z]',' ',df['post'][i])
    #review=df['post'][i]
    review=[ls.lemmatize(w) for w in word_tokenize(str(review).lower()) if w not in cstopwords]
    review=' '.join(review)
    text_corpus.append(review)
    
len(text_corpus)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cv=CountVectorizer()
X1=cv.fit_transform(text_corpus).toarray()
X1.shape
tfidfvec=TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')
X2=tfidfvec.fit_transform(text_corpus).toarray()
X2.shape
from sklearn.cluster import KMeans
from nltk.probability import FreqDist
from scipy.cluster import hierarchy as hc
hc.dendrogram(hc.linkage(X1,method='ward'))
km=KMeans(n_clusters=3)
km.fit(X1)
km.labels_
df1['labels']=km.labels_
df1['processed']=text_corpus
df1.head()
km.n_clusters
for i in range(km.n_clusters):
    df2=df1[df['labels']==i]
    df2=df2[['processed']]
    words=word_tokenize(str(list(set([a for b in df2.values.tolist() for a in b]))))
    dist=FreqDist(words)
    print('Cluster :',i)
    print('most common words :',dist.most_common(30))
text={}
for i,cluster in enumerate(km.labels_):
    oneDocument = df1['processed'][i]
    if cluster not in text.keys():
        text[cluster] = oneDocument
    else:
        text[cluster] += oneDocument
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk
_stopwords = set(stopwords.words('english') + list(punctuation)+["million","billion","year","millions","billions","y/y","'s","''","``"])
keywords = {}
counts={}
for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent=[word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100, freq, key=freq.get)
    counts[cluster]=freq
unique_keys={}
for cluster in range(3):   
    other_clusters=list(set(range(3))-set([cluster]))
    keys_other_clusters=set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique=set(keywords[cluster])-keys_other_clusters
    unique_keys[cluster]=nlargest(10, unique, key=counts[cluster].get)
unique_keys
