import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

import nltk

import os

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))



# Any results you write to the current directory are saved as output.
path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
list(embeddings['modi'][:5])
pd.Series(embeddings['modi'][:5])
embeddings.most_similar('modi',topn=10)
url = 'https://bit.ly/2S2yXEd'

data = pd.read_csv(url)

data.head()
doc1 = data.iloc[0,0]

print(doc1)

print(nltk.word_tokenize(doc1.lower()))
docs = data['review']

docs.head()
words = nltk.word_tokenize(doc1.lower())

temp = pd.DataFrame()

for word in words:

    try:

        print(word,embeddings[word][:5])

        temp = temp.append(pd.Series(embeddings[word][:5]),ignore_index=True)

    except:

        print(word,'is not there')
temp
docs = docs.str.lower().str.replace('[^a-z ]','')

docs.head()
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stopwords  = nltk.corpus.stopwords.words('english')



def clean_doc(doc):

    words = doc.split(' ')

    words_clean = [word for word in words if word not in stopwords]

    doc_clean= ' '.join(words_clean)

    return doc_clean



docs_clean = docs.apply(clean_doc)

docs_clean.head()
docs_clean.shape
docs_vectors =  pd.DataFrame()



for doc in docs_clean:

    words = nltk.word_tokenize(doc)

    temp =  pd.DataFrame()

    for word in words:

        try:

            word_vec = embeddings[word]

            temp = temp.append(pd.Series(word_vec),ignore_index=True)

        except:

            pass

    docs_vectors=docs_vectors.append(temp.mean(),ignore_index=True)   

docs_vectors.shape    
docs_vectors.head()
pd.isnull(docs_vectors).sum(axis=1).sort_values(ascending=False).head()
X = docs_vectors.drop([64,590])

Y = data['sentiment'].drop([64,590])
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=100)
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators=800)

model.fit(xtrain,ytrain)

test_pred =  model.predict(xtest)

accuracy_score(ytest,test_pred)
model = AdaBoostClassifier(n_estimators=800)

model.fit(xtrain,ytrain)

test_pred =  model.predict(xtest)

accuracy_score(ytest,test_pred)
url = 'https://bit.ly/2W21FY7'

data = pd.read_csv(url)

data.shape
data.head()
docs = data.loc[:,'Lower_Case_Reviews']

print(docs.shape)

docs.head()
Y = data['Sentiment_Manual']

Y.head()
Y.value_counts()
docs = docs.str.lower().str.replace('[^a-z ]','')

docs.head()
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stopwords  = nltk.corpus.stopwords.words('english')



def clean_doc(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    doc_clean= ' '.join(words_clean)

    return doc_clean



docs_clean = docs.apply(clean_doc)

docs_clean.head()
X = docs_clean 

X.shape,Y.shape
from sklearn.model_selection import train_test_split



xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=100)
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(min_df=5)

cv.fit(X)
XTRAIN = cv.transform(xtrain)

XTEST = cv.transform(xtest)
XTRAIN = XTRAIN.toarray()

XTEST = XTEST.toarray()
from sklearn.tree import DecisionTreeClassifier as dtc

from sklearn.metrics import accuracy_score

model = dtc(max_depth=10)

model.fit(XTRAIN,ytrain)

yp= model.predict(XTEST)

accuracy_score(ytest,yp)
from sklearn.naive_bayes import MultinomialNB as mnb

m1=mnb()

m1.fit(XTRAIN,ytrain)

yp1=m1.predict(XTEST)

accuracy_score(ytest,yp1)
from sklearn.naive_bayes import BernoulliNB as bnb

m2=bnb()

m2.fit(XTRAIN,ytrain)

yp2=m2.predict(XTEST)

accuracy_score(ytest,yp2)
from sklearn.feature_extraction.text import TfidfVectorizer



tv = TfidfVectorizer(min_df=5)

tv.fit(X)
XTRAIN = tv.transform(xtrain)

XTEST = tv.transform(xtest)
XTRAIN = XTRAIN.toarray()

XTEST = XTEST.toarray()
from sklearn.naive_bayes import MultinomialNB as mnb

mod=mnb()

mod.fit(XTRAIN,ytrain)

ypred=mod.predict(XTEST)

accuracy_score(ytest,ypred)
stopwords  = nltk.corpus.stopwords.words('english')



def clean_doc(doc):

    words = doc.split(' ')

    words_clean = [word for word in words if word not in stopwords]

    doc_clean= ' '.join(words_clean)

    return doc_clean



docs_clean = docs.apply(clean_doc)

docs_clean.head()
docs_vectors =  pd.DataFrame()



for doc in docs_clean:

    words = nltk.word_tokenize(doc)

    temp =  pd.DataFrame()

    for word in words:

        try:

            word_vec = embeddings[word]

            temp = temp.append(pd.Series(word_vec),ignore_index=True)

        except:

            pass

    docs_vectors=docs_vectors.append(temp.mean(),ignore_index=True)   

docs_vectors.shape    
Y.shape
df = pd.concat([docs_vectors,Y],axis=1)

df.head(3)
df[df.iloc[:,0].isnull()].shape
df = df.dropna(axis=0)
df.shape
X = df.drop(['Sentiment_Manual'],axis=1)

Y = df['Sentiment_Manual']
X.shape,Y.shape
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=.2,random_state=100)
xtrain.shape,ytrain.shape
from sklearn.tree import DecisionTreeClassifier as dtc

from sklearn.metrics import accuracy_score

model = dtc(max_depth=10)

model.fit(xtrain,ytrain)

yp= model.predict(xtest)

accuracy_score(ytest,yp)
data.head()
data.Sentiment_Manual.shape
docs_clean.shape
from nltk.sentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()



def get_sentiment(sentence,analyser=analyser):

    score = analyser.polarity_scores(sentence)['compound']

    if score > 0:

        return 1

    else:

        return 0