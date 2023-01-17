import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
df.head(10)
len(df)
df.describe().T
df.info()
df.isnull().sum()  #To check for Null Value 
df.groupby('sentiment').count()
df['sentiment'].value_counts().plot(kind='pie',autopct="%1.0f%%")
Length_of_ds=pd.DataFrame(df['review'].apply(len))
Length_of_ds.head(10)
Length_of_ds.describe()
df['review'][5]
import re

import nltk

from nltk.corpus import stopwords                   

from nltk.stem.porter import PorterStemmer            #For Stemming Of words

from bs4 import BeautifulSoup                         #To remove HTML Tags
corpus_bag=[]
features=df.iloc[:,0]
features
labels=df.iloc[:,1]
labels
df['review'][0]
def clean_html(text):

    soup=BeautifulSoup(text,"html.parser")

    return soup.get_text()
df['review']=df['review'].apply(clean_html)
def clean_mess(text):

    return re.sub('[^a-zA-Z]'," ",text)

    
df['review']=df['review'].apply(clean_mess)
df['review'][1]
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cv=CountVectorizer(ngram_range=(2,2),stop_words=stopwords.words('english'))

tfidft=TfidfTransformer()

tfidfv=TfidfVectorizer()

mnb=MultinomialNB()

logm=LogisticRegression()

#svm_mod=SVC()

rfc=RandomForestClassifier()

cv.fit_transform(features)
data1=cv.fit_transform(features)
data2=tfidft.fit_transform(data1)
X_train, X_test, y_train, y_test = train_test_split(data2, labels, test_size=0.3, random_state=0)
mnb.fit(X_train,y_train)
mnb.predict(X_test)
pred1=mnb.predict(X_test)
print(classification_report(y_test,pred1))
print('The Acuuracy Score : ',round(accuracy_score(y_test,pred1),2))
tfidfv_cust=TfidfVectorizer(ngram_range=(2,2),stop_words=stopwords.words('english'))
data11=tfidfv_cust.fit_transform(features)
X1_train, X1_test, y1_train, y1_test = train_test_split(data11, labels, test_size=0.2, random_state=0)
mnb.fit(X1_train,y1_train)
mnb.predict(X1_test)
pred11=mnb.predict(X1_test)
print(classification_report(y1_test,pred11))
print(confusion_matrix(y1_test,pred11))
print('The Acuuracy Score : ',round(accuracy_score(y1_test,pred11),2))
df_avc=pd.DataFrame({"Actual Class":y1_test,"Predicted Class":pred11})

df_avc.head(10)