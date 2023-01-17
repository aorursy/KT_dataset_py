import pandas as pd

import spacy

from spacy.lang.en import English

from spacy.lang.en.stop_words import STOP_WORDS
#Read the data set

df= pd.read_csv('../input/amazon-alexa/amazon_alexa.tsv', sep='\t')
#First five observations

df.head()
#Shape of the dataset

df.shape
#Count of output feature

df['feedback'].value_counts()
#Create list of punctuation marks

import string

punctuations=string.punctuation



#Create list of stopwords

nlp=spacy.load('en')

stop_words= spacy.lang.en.stop_words.STOP_WORDS



#Load English tokenizer, tagger, parser, NER and word vectors

parser=English()



#Create tokenizer function

def spacy_tokenizer(sentence):

  #Creating our token object, which is used to create documents with linguistic annotations.

  mytokens=parser(sentence)

  #Lemmatizing each token and converting each token into lowercase

  mytokens=[word.lemma_.lower().strip() if word.lemma_!='-PRON-' else word.lower_ for word in mytokens]

  #Removing stopwords

  mytokens= [word for word in mytokens if word not in stop_words and word not in punctuations]

  #Return preprocessed list of tokens

  return mytokens
#To further clean our text data, we’ll also want to create a custom transformer for removing initial and end spaces and converting text into lower case.

from sklearn.base import TransformerMixin

class predictors(TransformerMixin):

  def transform(self,X, **transform_params):

    return [clean_text(text) for text in X]

  

  def fit(self,X,y=None, **fit_params):

    return self

  

  def get_params(self, deep=True):

    return {}

#Basic function to clean the text

def clean_text(text):

  #Removing spaces and converting text into lowercase

  return text.strip().lower()
#We need to perform feature vectorization in order to transform word tokens into numbers. We can do it using tf-idf ,Bag of Words.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



bow_vector= CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,3))

tfidf_vector= TfidfVectorizer(tokenizer=spacy_tokenizer)
#Spliting data into train and test

from sklearn.model_selection import train_test_split

X= df['verified_reviews']

y=df['feedback']

X_train, X_test,y_train,y_test= train_test_split(X,y,test_size=0.25)
#We will use pipeline to integrate the entire modeling technique.

#Pipeline 1

#Decision Tree with GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

classifier= DecisionTreeClassifier(class_weight='balanced')

parameters={'min_samples_split': range(10,500,20),'max_depth': range(1,20,2)}

clf=GridSearchCV(classifier,parameters)
#Pipeline using tfidf

pipe1= Pipeline([('cleaner', predictors()),

                 ('vectorizer',tfidf_vector),

                 ('classifier', classifier)])

pipe1.fit(X_train,y_train)
#Evaluation of GridSearchCV Decision Tree model

from sklearn import metrics

predicted= pipe1.predict(X_test)

print('Accuracy:',metrics.accuracy_score(predicted,y_test))

print('Precision:',metrics.precision_score(predicted,y_test))

print('Recall:',metrics.recall_score(predicted,y_test))



import seaborn as sns

import matplotlib.pyplot as plt

cfm=metrics.confusion_matrix(y_test,predicted)

lbl1=['Predicted Negetive', 'Predicted Positive']

lbl2=['Actual Negetive', 'Actual Positive']

sns.heatmap(cfm, annot=True, cmap='Blues',fmt='d',xticklabels=lbl1,yticklabels=lbl2)

plt.show()
#Pipeline 2

#Decision Tree without GridSearchCV

pipe2= Pipeline([('cleaner',predictors()),

                 ('vectorizer',bow_vector),

                 ('classifier',clf)])

pipe2.fit(X_train,y_train)
##Evaluation of simple Decision Tree model

predicted2= pipe2.predict(X_test)

print('Accuracy:',metrics.accuracy_score(predicted2,y_test))

print('Precision:',metrics.precision_score(predicted2,y_test))

print('Recall:',metrics.recall_score(predicted2,y_test))



cfm=metrics.confusion_matrix(y_test,predicted2)

lbl1=['Predicted Negetive', 'Predicted Positive']

lbl2=['Actual Negetive', 'Actual Positive']

sns.heatmap(cfm, annot=True, cmap='Blues',fmt='d',xticklabels=lbl1,yticklabels=lbl2)

plt.show()
#Pipeline 3

#SVM with GridSearchCV

from sklearn.svm import SVC

svc=SVC(class_weight='balanced')

params1={'kernel':['linear','rbf','poly','sigmoid'],'C':[0.01,0.1,1,10],'gamma':[0.01,0.1,1,10]}

gs_svc=GridSearchCV(svc,params1)
#Pipeline using tfidf

pipe3= Pipeline([('cleaner', predictors()),

                 ('vectorizer',tfidf_vector),

                 ('classifier', gs_svc)])

pipe3.fit(X_train,y_train)
#Evaluation of GridSearchCV SVM

predicted3= pipe3.predict(X_test)

print('Accuracy:',metrics.accuracy_score(predicted3,y_test))

print('Precision:',metrics.precision_score(predicted3,y_test))

print('Recall:',metrics.recall_score(predicted3,y_test))



cfm=metrics.confusion_matrix(y_test,predicted3)

lbl1=['Predicted Negetive', 'Predicted Positive']

lbl2=['Actual Negetive', 'Actual Positive']

sns.heatmap(cfm, annot=True, cmap='Blues',fmt='d',xticklabels=lbl1,yticklabels=lbl2)

plt.show()