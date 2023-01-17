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
#Import necessary libraries

import numpy as np

import pandas as pd

import string

import nltk

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

from matplotlib import pyplot as plt
df=pd.read_csv("../input/abcnews-date-text.csv")

df.head()
text=pd.DataFrame()

text['Text']=df['headline_text']

text.head()

#Convert all characters to lower case

text['clean_text']=text['Text'].str.lower()

text.head()

#Apply regular expressions to retain only alphabets, #, and spaces

text['clean_text']=text['clean_text'].str.replace('[^a-z ]','')

text.head()

#Remove stopwords

stop=stopwords.words('english')

#creating function for stop words

def sw(text):

    text=[word for word in text.split() if word not in stop]

    return " ".join(text)

text['split_words']=text['clean_text'].apply(sw)

text.head()

#removing words less than 4 character   - optional

def lw(x):

    x=[word for word in x.split() if len(word)>3]

    return " ".join(x)

text['split_words']=text['split_words'].apply(lw)

text.head()
#creating tf-idf

tf_idf_vet=TfidfVectorizer()

score=tf_idf_vet.fit_transform(text['split_words'])

score
#building the LDA model - dividing DTM to 5 topics

# for Topic modeling 2 main libs= gensim and 

from sklearn.decomposition import LatentDirichletAllocation



#creating the lda model

lda_model=LatentDirichletAllocation(n_topics=10,random_state=1234,max_iter=15)



#fitting lda model on DTM(score)

lda_output= lda_model.fit_transform(score)
#Find the dominating topic for each document

#create the column for the document - topic matrix

topicnames=['Topic '+str(i) for i in range (lda_model.n_topics)]

print(topicnames)

#create the row name for the document - topic matrix

docnames=['Doc '+str(i) for i in range (len(text['split_words']))]
#create a dataframe for Document - Topic Matrix

df_document_topic=pd.DataFrame(np.round(lda_output,2),index=docnames,columns=topicnames)

#finding the dominating topic

dominating_topic=np.argmax(df_document_topic.values,axis=1)

dominating_topic

df_document_topic['Dominating_topic']=dominating_topic

df_document_topic



#group by

df_document_topic.groupby(['Dominating_topic']).size()
# creating  TTM - topic term matrix

df_topic_keywords=pd.DataFrame(lda_model.components_)



#Assigning the column and index

df_topic_keywords.columns=tf_idf_vet.get_feature_names()



df_topic_keywords.index=topicnames



df_topic_keywords
def show_topics(vectorizer=tf_idf_vet,model=lda_model,n_words=20):

    keywords=np.array(tf_idf_vet.get_feature_names())

    topic_keywords=[]

    for topic_weights in lda_model.components_:

        top_keyword_locs=(-topic_weights).argsort()[:n_words]

        topic_keywords.append(keywords.take(top_keyword_locs))

    return topic_keywords

    

topic_keywords=show_topics(vectorizer=tf_idf_vet,model=lda_model,n_words=20)

topic_keywords
#Creating topic word data frame

df_topic_keywords=pd.DataFrame(topic_keywords)



#giving the index name and column name

df_topic_keywords.columns=['Word '+str(i) for i in range(df_topic_keywords.shape[1])]

df_topic_keywords.index=['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

df_topic_keywords
# Adding the column of topic to our data frame



df['topic'] = dominating_topic

df.head()
#For each topic find the total number of documents.

df['topic'].value_counts()