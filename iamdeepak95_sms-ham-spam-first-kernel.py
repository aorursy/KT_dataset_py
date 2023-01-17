# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize



from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import *

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
#reading the file

sms=pd.read_table('../input/sms.tsv',header=None,names=['Label','Message'])

df_sms=pd.DataFrame(sms)

df_sms.head() #print the value of sms dataframe



#pie plot of labels

plt.pie(df_sms['Label'].value_counts(),shadow=True,autopct='%1.1f%%',labels=df_sms['Label'].value_counts().index,explode=[0,0.25])





#bar plot of the labels

sns.countplot(data=df_sms,x=df_sms.Label)



#data preprocessing

df_sms['sms_label']=df_sms.Label.map({'ham':0,'spam':1})   #mapping categorical variable

label=df_sms['sms_label']

sms=df_sms['Message']





#splitting dataset into training and test values

X_train,X_test,Y_train,Y_test=train_test_split(sms,label,random_state=1,test_size=0.25)

print(X_train.shape)

print(X_test.shape)



#initalizing count vectorizer

cnt_vect=CountVectorizer(ngram_range=(1,1),max_features=None)



#applying vectorizer on training and test data

cnt_vect.fit(X_train)

print(len(cnt_vect.vocabulary_))



X_train_dtm=cnt_vect.transform(X_train)

print(X_train_dtm.shape)

print(Y_train.shape)

print(Y_test.shape)



X_test_dtm=cnt_vect.transform(X_test)

print(X_test_dtm.shape)
#nitializig and applying Naive Bayes 



nb=MultinomialNB()

nb.fit(X_train_dtm,Y_train)

pred_y=nb.predict(X_test_dtm)

print("shape of predicted dataset", pred_y.shape)



m_confusion_test = confusion_matrix(Y_test, nb.predict(X_test_dtm))

pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],

            index = ['Actual 0', 'Actual 1'])

print("precision score")

print(precision_score(Y_test,pred_y))













#ROC curve

y_pred_proba=nb.predict_proba(X_test_dtm)[:,1]

roc_auc_score(Y_test, y_pred_proba)

fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba)

auc = roc_auc_score(Y_test, y_pred_proba)

plt.style.use('seaborn')

plt.plot(fpr,tpr,label="SMS - Spam detection, auc="+str(auc))

plt.legend(loc=4)
#plotting wordcloud for ham messages

X_ham  = df_sms[(df_sms.sms_label == 0)].Message

X_spam = df_sms[(df_sms.sms_label == 1)].Message



words_ham = ' '.join(X_ham)



wordcloud = WordCloud(width = 800, 

                     height = 800, 

                     background_color ='black', 

                     #stopwords = stopwords, 

                     min_font_size = 10).generate(words_ham) 

# plot the WordCloud image                        

plt.figure(figsize = (9, 12), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0)



#plotting wordcloud for spam messages

word_spam=''.join(X_spam)

wordcloud2=WordCloud(width=800,height=800,background_color='black',min_font_size=10).generate(word_spam)

plt.figure(figsize = (9, 12), facecolor = None) 

plt.imshow(wordcloud2) 

plt.axis("off") 

plt.tight_layout(pad = 0)