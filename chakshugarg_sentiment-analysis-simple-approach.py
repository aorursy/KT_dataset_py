#load - > clean - > remove punctuations - > remove stop words - > remove having length less then 2 - >  remove numeric - > count vectorizer - > multinomial NB - > predict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
import warnings
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

print('########Methods#########')
def readData(FileLocation,FileType):
    print('   ')
    print('loading file - ' + FileLocation)
    
    if FileType=='csv':
        data = pd.read_csv(FileLocation)
    
    print("Number of Columns - " + str(len(data.columns)))
    print("DataTypes - ")
    print(data.dtypes.unique())
    print(data.dtypes)
    print('checking null - ')
    print(data.isnull().any().sum(), ' / ', len(data.columns))
    print(data.isnull().any(axis=1).sum(), ' / ', len(data))
    print ('columns having null values - ')
    print(data.columns[data.isnull().any()])
    print(data.head())
    return data
    

def DataCleaning(data, columnsToBeDropped, fillNAValues):
    if (columnsToBeDropped):
        data = data.drop(columnsToBeDropped,axis=1)
    data.dropna(thresh=0.8*len(data), axis=1)
    data.dropna(thresh=0.8*len(data))
    if len(fillNAValues) > 1:
        for value in fillNAValues:
            data[value].fillna(data[value].median(), inplace = True)
    return data

def cleanDataForNLP(TextData):
    TextData = re.sub('[^A-Za-z]+', ' ', TextData)    
    word_tokens = word_tokenize(TextData)
    filteredText = ""
    for w in word_tokens:
        if w not in stop_words and len(w) > 2 and not w.isnumeric() and w not in string.punctuation:
            filteredText = filteredText + " " + w
    
    return filteredText.strip()    

def IspositiveRating(rating):
    if rating > 2:
        val = 1
    else:
        val = 0
    return val

print("========Loading Data========")
FileLocation="../input/Womens Clothing E-Commerce Reviews.csv"
FileType="csv"
data=readData(FileLocation,FileType)

print("========Cleaning Data========")
pd.set_option('display.expand_frame_repr', False)
columnsToBeDropped=['Unnamed: 0','Clothing ID','Title','Age','Title','Recommended IND','Positive Feedback Count','Division Name','Department Name','Class Name']
fillNAValues=['']
data=DataCleaning(data, columnsToBeDropped, fillNAValues)

stop_words = list(get_stop_words('en'))         
nltk_words = list(stopwords.words('english'))   
stop_words.extend(nltk_words)
data['Review Text']=data['Review Text'].astype(str).apply(cleanDataForNLP)
data['IsPositiveReview']=data['Rating'].apply(IspositiveRating)

x=data['Review Text']
y=data['IsPositiveReview']
transformer=CountVectorizer(analyzer='word').fit(x) 
x = transformer.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=101)
nb = MultinomialNB()
nb.fit(x_train, y_train)
predict=nb.predict(x_test)
print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))

