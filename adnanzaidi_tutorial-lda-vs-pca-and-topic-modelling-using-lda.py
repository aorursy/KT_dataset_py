import nltk

import numpy as np

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import *

from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import  CountVectorizer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import pandas as pd

from sklearn.datasets import load_iris

from sklearn.decomposition import PCA,LatentDirichletAllocation

import matplotlib.pyplot as plt
data=load_iris().data

target=load_iris().target

target_names=load_iris().target_names
dataframe=pd.DataFrame(data=np.concatenate((data,target.reshape(150,1)),axis=1),columns=['col_1','col_2','col_3','col_4','target'])
dataframe.head()
dataframe.drop(columns=['target'],axis=1,inplace=True)
pca = PCA (n_components=2)

X_feature_reduced = pca.fit(dataframe).transform(dataframe)
print ('First component explain {} variance of data and second component explain {} variance of data'.format(pca.explained_variance_ratio_[0],pca.explained_variance_ratio_[1]))
plt.scatter(X_feature_reduced[:,0],X_feature_reduced[:,1],c=target)

plt.title("PCA")

plt.show()
lda = LatentDirichletAllocation(n_components=2)

X_feature_reduced = lda.fit(dataframe).transform(dataframe)
plt.scatter(X_feature_reduced[:,0],X_feature_reduced[:,1],c=target)

plt.title('LDA')

plt.show()
lemmatizer=WordNetLemmatizer() #For words Lemmatization

stop_words=set(stopwords.words('english'))
def TokenizeText(text):

    ''' 

     Tokenizes text by removing various stopwords and lemmatizing them

    '''

    text=re.sub('[^A-Za-z0-9\s]+', '', text)

    word_list=word_tokenize(text)

    word_list_final=[]

    for word in word_list:

        if word not in stop_words:

            word_list_final.append(lemmatizer.lemmatize(word))

    return word_list_final
def gettopicwords(topics,cv,n_words=10):

    '''

        Print top n_words for each topic.

        cv=Countvectorizer

    '''

    for i,topic in enumerate(topics):

        top_words_array=np.array(cv.get_feature_names())[np.argsort(topic)[::-1][:n_words]]

        print ("For  topic {} it's top {} words are ".format(str(i),str(n_words)))

        combined_sentence=""

        for word in top_words_array:

            combined_sentence+=word+" "

        print (combined_sentence)

        print (" ")
import os

os.listdir("../input/million-headlines/")
df=pd.read_csv('../input/million-headlines/abcnews-date-text.csv',usecols=[1])
df.head()
%%time 

num_features=100000

cv=CountVectorizer(tokenizer=TokenizeText,max_features=num_features,ngram_range=(1,2))

transformed_data=cv.fit_transform(df['headline_text'])
%%time

no_topics=10  ## We can change this, hyperparameter

lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(transformed_data)
gettopicwords(lda.components_,cv)
docs=df['headline_text'][:10]
data=[]

for doc in docs:

    data.append(lda.transform(cv.transform([doc])))
cols=['topic'+str(i) for i in range(1,11)]

doc_topic_df=pd.DataFrame(columns=cols,data=np.array(data).reshape((10,10)))
doc_topic_df['major_topic']=doc_topic_df.idxmax(axis=1)

doc_topic_df['raw_doc']=docs
doc_topic_df.head()
print ("Complete")