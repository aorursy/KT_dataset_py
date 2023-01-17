# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import random

import spacy



from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from pandas.io.json import json_normalize

from spacy import displacy



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report





nlp = spacy.load('en_core_web_sm')    

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

sns.set_style({'xtick.bottom':False,

               'ytick.left':False,

               'axes.spines.bottom':False,

               'axes.spines.top':False,

               'axes.spines.left':False,

               'axes.spines.right':False

              })



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_json('../input/Sarcasm_Headlines_Dataset.json',lines=True)

data.head()
data.shape
data.info()
data.isnull().sum()
def explore_headline(text):    

    doc=nlp(text)

    print(f'Headline : {doc}')

    print(f'\nTotal number of tokens : {len(doc)} \n')



    for token in doc:

        print(token.text,end=' | ')

        

    print('\n')

    

    for token in doc:

        print(f'{token.text:{12}}{token.pos_:{10}}{token.dep_:{12}}{str(spacy.explain(token.dep_))}')

    

    print(f'\nTotal number of Sentences : {len(list(doc.sents))}')

    for sent in doc.sents:

        print(sent)

        

    if len(doc.ents)>0:

        print(f'\nTotal number of Entity : {len(doc.ents)}\n')    

        for ent in doc.ents:

             print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))

        displacy.render(doc,style='ent',jupyter=True)

    

    displacy.render(doc,style='dep',jupyter=True,options={'distance': 80}) 

    

def get_ents(text):

    doc=nlp(text)

    return len(doc.ents)

def get_tokens(text):

    doc=nlp(text)

    return len(doc)

def get_sents(text):

    doc=nlp(text)

    return len(list(doc.sents))
explore_headline(data['headline'][24883])
explore_headline(data['headline'][6089])
explore_headline(data['headline'][7667])
data_sample = data.sample(frac=.30,random_state=1)

data_sample.head()
explore_headline(data_sample['headline'][8304])
data_sample['ents_num'] = data_sample['headline'].apply(get_ents)

data_sample['tokens_num'] = data_sample['headline'].apply(get_tokens)

data_sample['sents_num'] = data_sample['headline'].apply(get_sents)



data_sample.head()
fig,(ax,ax1,ax2)=plt.subplots(nrows=3,ncols=1,figsize=(15,15))

sns.countplot(x='ents_num',data=data_sample,hue='is_sarcastic',ax=ax,palette='spring')

sns.countplot(x='tokens_num',data=data_sample,hue='is_sarcastic',ax=ax1,palette='winter')

sns.countplot(x='sents_num',data=data_sample,hue='is_sarcastic',ax=ax2,palette='cool')
data_sample.drop(['article_link','ents_num','tokens_num','sents_num'],axis=1,inplace=True)

data.drop(['article_link'],axis=1,inplace=True)
blanks = []

for i,he,is_sa in data_sample.itertuples():

    if type(he) == str:

        if he.isspace():

            blanks.append(i)

print(len(blanks), 'blanks: ', blanks)
#train_test_split

X = data_sample['headline']

y = data_sample['is_sarcastic']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Building a pipeline

#Naive Bayes

classifier_nb = Pipeline([('tfidf',TfidfVectorizer()),

                     ('clf',MultinomialNB())])



#Logistic regression

classifier_lr = Pipeline([('tfidf',TfidfVectorizer()),

                     ('clf',LogisticRegression(solver='saga'))])



#Random Forest

classifier_rf = Pipeline([('tfidf',TfidfVectorizer()),

                     ('clf',RandomForestClassifier(bootstrap= False, criterion= 'entropy', n_estimators= 100))])



#Linear SVC

classifier_svc = Pipeline([('tfidf',TfidfVectorizer()),

                     ('clf',LinearSVC())])
#Feeding the data

classifier_nb.fit(X_train,y_train)

classifier_lr.fit(X_train,y_train)

classifier_rf.fit(X_train,y_train)

classifier_svc.fit(X_train,y_train)
#Predicting 

y_pred = classifier_nb.predict(X_test)

yt_pred = classifier_nb.predict(X_train)

#Analyzing

cm = confusion_matrix(y_test,y_pred)

print(f'Confusion Matrix :\n {cm}\n')

print(f'Test Set Accuracy Score :\n {accuracy_score(y_test,y_pred)}\n')

print(f'Train Set Accuracy Score :\n {accuracy_score(y_train,yt_pred)}\n')

print(f'Classification Report :\n {classification_report(y_test,y_pred)}')
#Predicting 

y_pred = classifier_lr.predict(X_test)

yt_pred = classifier_lr.predict(X_train)

#Analyzing

cm = confusion_matrix(y_test,y_pred)

print(f'Confusion Matrix :\n {cm}\n')

print(f'Test Set Accuracy Score :\n {accuracy_score(y_test,y_pred)}\n')

print(f'Train Set Accuracy Score :\n {accuracy_score(y_train,yt_pred)}\n')

print(f'Classification Report :\n {classification_report(y_test,y_pred)}')
y_pred = classifier_rf.predict(X_test)

yt_pred = classifier_rf.predict(X_train)



cm = confusion_matrix(y_test,y_pred)

print(f'Confusion Matrix :\n {cm}\n')

print(f'Test Set Accuracy Score :\n {accuracy_score(y_test,y_pred)}\n')

print(f'Train Set Accuracy Score :\n {accuracy_score(y_train,yt_pred)}\n')

print(f'Classification Report :\n {classification_report(y_test,y_pred)}')
y_pred = classifier_svc.predict(X_test)

yt_pred = classifier_svc.predict(X_train)



cm = confusion_matrix(y_test,y_pred)

print(f'Confusion Matrix :\n {cm}\n')

print(f'Test Set Accuracy Score :\n {accuracy_score(y_test,y_pred)}\n')

print(f'Train Set Accuracy Score :\n {accuracy_score(y_train,yt_pred)}\n')

print(f'Classification Report :\n {classification_report(y_test,y_pred)}')