import numpy as np 

import pandas as pd 

import matplotlib as mp

import os

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD

import matplotlib.pyplot as plt
#uploading data in dataframe

train=pd.read_csv("../input/train.csv",sep=',')
#displaying exemple data

train.head(5)
#displaying exemple of insincere data 

train[train.target==1].head(5)
#displayin dataframe info

train.info()
#counting target values

train.target.value_counts()
train['word_count'] = train['question_text'].apply(lambda x: len(str(x).split(" ")))

#basic statistic about word_count

train.word_count.describe()
#lower case

train['question_text'] = train['question_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#Removing Punctuation

train['question_text'] = train['question_text'].str.replace('[^\w\s]','')

#Removing numbers

train['question_text'] = train['question_text'].str.replace('[0-9]','')

#Remooving stop words and words with length <=2

from nltk.corpus import stopwords

stop = stopwords.words('english')

train['question_text'] = train['question_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop and len(x)>2))

#Stemming

#from nltk.stem import SnowballStemmer

#ss=SnowballStemmer('english')

#train['question_text'] = train['question_text'].apply(lambda x: " ".join(ss.stem(x) for x in x.split()))

from nltk.stem import WordNetLemmatizer

wl = WordNetLemmatizer()

train['question_text'] = train['question_text'].apply(lambda x: " ".join(wl.lemmatize(x,'v') for x in x.split()))
from nltk.stem import SnowballStemmer,WordNetLemmatizer,PorterStemmer,LancasterStemmer

wl = WordNetLemmatizer()

ss=SnowballStemmer('english')

ps=PorterStemmer()

ls=LancasterStemmer()

test_list=['does','peaople','writing','beards','enjoyment','bought','leaves','gave','given','generaly','would']

for item in test_list :

    print('lemmatizer : %s'%wl.lemmatize(item,'v'))

    print('SS stemmer : %s'%ss.stem(item))

    print('PS stemmer : %s'%ps.stem(item))

    print('LS stemmer : %s'%ls.stem(item))

train.head(5)
tfidf_v = TfidfVectorizer(min_df=20,max_df=0.8,sublinear_tf=True,ngram_range={1,2})

#matrixTFIDF= tfidf_v.fit_transform(train.question_text)

matrixTFIDF= tfidf_v.fit_transform(train[train.target==1].question_text)
print(matrixTFIDF.shape)
svd=TruncatedSVD(n_components=15, n_iter=10,random_state=42)

X=svd.fit_transform(matrixTFIDF)             
def get_topics(components, feature_names, n=15):

    for idx, topic in enumerate(components):

        print("Topic %d:" % (idx))

        print([(feature_names[i], topic[i])

                        for i in topic.argsort()[:-n - 1:-1]])
get_topics(svd.components_,tfidf_v.get_feature_names())