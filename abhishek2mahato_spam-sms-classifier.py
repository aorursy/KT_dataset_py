import pandas as pd

import numpy as np

import spacy

from spacy.lang.en import English

from spacy.lang.en.stop_words import STOP_WORDS
#Loading the dataset

df = pd.read_csv('../input/spamclass/Spam SMS Collection', sep='\t', names=['label', 'message'])
#First five observations

df.head()
#Shape of the dataset

df.shape
#Count of output feature

df['label'].value_counts()
#Create list of punctuation marks

import string

punctuations=string.punctuation



#Create list of stopwords

nlp=spacy.load('en')

stop_words=spacy.lang.en.stop_words.STOP_WORDS



#Load English tokenizer, tagger, parser, NER and word vectors

parser=English()



#Create tokenizer function

def spacy_tokenizer(sentence):

  #Creating our token object, which is used to create documents with linguistic annotations.

  mytokens=parser(sentence)

  #Lemmatizing each token and converting each token into lowercase

  mytokens=[word.lemma_.lower().strip() if word.lemma_!='-PRON' else word.lower_ for word in mytokens]

  #Removing stopwords

  mytokens=[word for word in mytokens if word not in stop_words and word not in punctuations]

  #Return preprocessed list of tokens

  return mytokens
#To further clean our text data, we’ll also want to create a custom transformer for removing initial and end spaces and converting text into lower case.

from sklearn.base import TransformerMixin

class predictors(TransformerMixin):

  def transform(self,X, **transform_params):

    return[clean_text(text)for text in X]



  def fit(self,X,y=None, **fir_params):

    return self

  

  def get_params(self,deep=True):

    return {}

#Basic function to clean the text

def clean_text(text):

  #Removing spaces and converting text into lowercase

  return text.strip().lower()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



bow_vector=CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,3))

tfidf_vector= TfidfVectorizer(tokenizer=spacy_tokenizer)
from sklearn.model_selection import train_test_split

X=df['message']

y=pd.get_dummies(df['label'])

y=y.drop(['spam'],1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

svc=SVC(class_weight='balanced')

params={'kernel':['linear','rbf','poly','sigmoid'],'C':[0.01,0.1,1,10],'gamma':[0.01,0.1,1,10]}

gs_svc=GridSearchCV(svc,params)
pipe= Pipeline([('cleaner', predictors()),

                 ('vectorizer',tfidf_vector),

                 ('classifier', gs_svc)])

pipe.fit(X_train,y_train)
from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

predicted=pipe.predict(X_test)

print('Accuracy:',metrics.accuracy_score(predicted,y_test))

print('Precision:',metrics.precision_score(predicted,y_test))

print('Recall:',metrics.recall_score(predicted,y_test))



cfm=metrics.confusion_matrix(y_test,predicted)

lbl1=['Predicted Negetive', 'Predicted Positive']

lbl2=['Actual Negetive', 'Actual Positive']

sns.heatmap(cfm, annot=True, cmap='Blues',fmt='d',xticklabels=lbl1,yticklabels=lbl2)

plt.show()