import os
import csv 
from os import path
import pandas as pd
import numpy as np
from pandas import DataFrame
from datetime import datetime
from time import mktime
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import emoji
import re
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk import classify
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import style
%matplotlib inline
import wordcloud   # Sentiment-based Word Clouds
from wordcloud import WordCloud, STOPWORDS 
from PIL import Image
print(os.listdir("../input"))
df=pd.read_csv('../input/labeledTrainData.tsv',header=0,sep='\t', error_bad_lines=True,encoding='utf8')
df=df[0:1000]
print(df.shape)
df.head()
def sc(x):
    score=SentimentIntensityAnalyzer().polarity_scores(x)
    return score['compound']
df["SentScore"]=df["review"].map(sc)
    
df.head()
def sca(lb):
    if lb >= .6:
        return "Very Good"
    elif (lb > .2) and (lb < .6):
        return "Good"
    elif (lb > -.2) and (lb < .2):
        return "Average"
    elif (lb > -.6) and (lb < -.2):
        return "Disappointing"
     
    else:
        return "Regrettable"
df["SentClass"]=df["SentScore"].map(sca)
def scan(lb):
    if lb >= .6:
        return 5
    elif (lb > .2) and (lb < .6):
        return 4
    elif (lb > -.2) and (lb < .2):
        return 3
    elif (lb > -.6) and (lb < -.2):
        return 2
     
    else:
        return 1
df["Rating"]=df["SentScore"].map(scan)
df.head(15)
# Make text lower case
df["review"]  = df["review"].str.lower()
# Remove digits from text
def Remove_digit(text):
    result = re.sub(r"\d", "", text)
    return result
df["review"]  = df["review"].apply(Remove_digit)
# Remove HTML from text
def remove_html(text):
    result = re.sub(r'<.*?>','',text) # Find out anything that is in between < & > symbol 
    return result
df["review"]  = df["review"].apply(remove_html)
# Remove special text characters
def remove_spl(text):
    result = re.sub(r'\W',' ',text) 
    return result
df["review"]  = df["review"].apply(remove_spl)
def lem_word(text):
    result= WordNetLemmatizer().lemmatize(text)
    return result
df["review"]  = df["review"].apply(lem_word)
df.head()
corpus=df["review"].tolist()
len(corpus)
corpus[0:2]
# Count Vectorisation

cv = text.CountVectorizer(input=corpus,stop_words ='english', ngram_range=(1,3))
matrix = cv.fit_transform(corpus)
X=pd.DataFrame(matrix.toarray(), columns=cv.get_feature_names())
X.shape
y = df['Rating']
len(y)
# We are going to try and run the RandomForest Classifier on X=review and y=non-continuous sentiment
# using the fit transformation of the tf - idf matrix to array =X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=200)
len(X_train)
from sklearn.ensemble import RandomForestClassifier
text_classifier=RandomForestClassifier(n_estimators=100, random_state=200)
text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
predictionsn = text_classifier.predict(X_train)
predictionsn[0:30]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))
svd=decomposition.TruncatedSVD(n_components=300)
matrix_reduced_svd=svd.fit_transform(matrix)
svd.components_
svd.explained_variance_ratio_
sum(svd.explained_variance_ratio_)
pca=decomposition.PCA(10)
scalers=preprocessing.StandardScaler()
X1=scalers.fit_transform(matrix_reduced_svd)
pca_mod=pca.fit_transform(X1)
pca.components_  #loadings
pca.explained_variance_ratio_ #Eigenvalues
pd.Series(pca.explained_variance_ratio_).plot(kind='bar', color = "orange") #Eigenvalues
plt.show()
# TF-IDF, Term Frequency and Inverse Document Freq
tf = text.TfidfVectorizer(input=corpus,stop_words ='english', ngram_range=(1,3))
matrix5 = tf.fit_transform(corpus)
X=pd.DataFrame(matrix5.toarray(), columns=tf.get_feature_names())
X.shape
# We are going to try and run the RandomForest Classifier on X=review and y=non-continuous sentiment
# using the fit transformation of the tf - idf matrix to array =X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=200)
from sklearn.ensemble import RandomForestClassifier
text_classifier=RandomForestClassifier(n_estimators=100, random_state=200)
text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
predictionsn = text_classifier.predict(X_train)
predictionsn[0:30]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))
# Checking the predictions on train and test data
result_train=pd.DataFrame({'Y_train':y_train, 'Predictions':predictionsn})
result_train.head()
result_test=pd.DataFrame({'Y_test':y_test, 'Predictions':predictions})
result_test[20:40]
# Using Naive Bayes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=200)
clf = MultinomialNB().fit(X_train, y_train)
predictions = clf.predict(X_test)
predictionsn = clf.predict(X_train)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))
# Checking the predictions on train and test data
result_train=pd.DataFrame({'Y_train':y_train, 'Predictions':predictionsn})
result_train[30:50]
result_test=pd.DataFrame({'Y_test':y_test, 'Predictions':predictions})
result_test[90:120]
# Using SVM classifier
textclf=SGDClassifier(penalty='l2',alpha=1e-3, random_state=42)
textclf = textclf.fit(X_train, y_train)
predictions = textclf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))
# Checking the predictions on train and test data
result_train=pd.DataFrame({'Y_train':y_train, 'Predictions':predictionsn})
result_train[30:50]
result_test=pd.DataFrame({'Y_test':y_test, 'Predictions':predictions})
result_test[90:120]