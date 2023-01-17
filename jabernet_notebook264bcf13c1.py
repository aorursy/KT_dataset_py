#Start with the basic imports
import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import os
import numpy as np
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
from textblob import TextBlob
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import math
from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')
import keras
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import (LSTM, 
                          Embedding, 
                          BatchNormalization,
                          Dense, 
                          TimeDistributed, 
                          Dropout, 
                          Bidirectional,
                          Flatten, 
                          GlobalMaxPool1D)
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    accuracy_score
)

#Importing the data
data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
submission = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
#Basic processing

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['length'] = data['text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['text'].apply(lambda x: count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

#I want to make my own normalizer for tweet length


def normalize(numbers):
    X_bar = numbers.mean()
    sigma = numbers.std()
    standardized = []
    for i in numbers:
        z = (X_bar - i)/sigma
        standardized.append(norm.cdf(z))
        
    return pd.Series(standardized)

data['Normalized_length'] = normalize(data['length'])
#Now let's look at "keyword"
data['keyword'].isnull().sum()/len(data['keyword'])*100

data['keyword'] = data['keyword'].fillna(data['keyword'].mode().iloc[0])
#Let's turn keywords into numeric values:
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['keyword']=encoder.fit_transform(data['keyword'])

data['keyword'].value_counts()
#Tweet sentiment and polarity
#Textblob for sentiment analysis
sent = []
pol = []
for tweet in data['text']:
    tb = TextBlob(tweet)
    pol.append(tb.sentiment[0])
    sent.append(tb.sentiment[1])

sent = pd.Series(sent)
pol = pd.Series(pol)
data['Sentiment']= sent
data['Polarity']= pol

#Let's see what we are dealing with here:
data.head()
data.shape
#See what the distribution of 'target' is:
Target = data['target'].value_counts()
print(Target)
print("Percent Fake",round(Target[0]/ (Target[1]+Target[0]),3)*100)
print("Percent Real",round(Target[1]/ (Target[1]+Target[0]),3)*100)
#A graphical plot of target:
sns.barplot(Target.index,Target.values,alpha=0.8)
plt.title('Fake/ Not fake')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Tweets', fontsize=12)
plt.show()
#Tweet character length:
#Check distribution of "length"
sns.distplot(data['length'], bins=20, kde=False, rug=True)
#Getting a ratio of fake/not fake by character length
def cal_ratio(x):
    n_1 = sum(x['target'].values == 1)
    n_0 = sum(x['target'].values == 0)
    return n_1/n_0
twl = data.groupby('length').apply(cal_ratio)
print(twl)
#Apparently I will need to bin this!
binned_len = []
for i in data['length']:
    if i < 25:
        binned_len.append(25)
    elif i >= 25 and i < 50:
        binned_len.append(50)
    elif i >= 50 and i < 75:
        binned_len.append(75)
    elif i >= 75 and i < 100:
        binned_len.append(100)
    elif i >= 100 and i < 125:
        binned_len.append(125)
    elif i >= 125 and i < 150:
        binned_len.append(150)
Binned_len = pd.Series(binned_len)
data['Binned_len'] = Binned_len

twl = data.groupby('Binned_len').apply(cal_ratio)
print(twl)
#Let's graph that!
sns.barplot(twl.index,twl.values,alpha=1)
plt.title('fake/ not fake')
plt.ylabel('Ratio', fontsize=12)
plt.xlabel('Character length', fontsize=12)
plt.show()
#Check distribution of percentage of characters that are punctuaation
sns.distplot(data['punct%'], bins=20, kde=False, rug=True)
#Sentiment
sns.distplot(data['Sentiment'])
#Polarity
sns.distplot(data['Polarity'])

X = data[["keyword","Sentiment", "Polarity", "Normalized_length", "punct%","Sentiment","Polarity"]]
y = data['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
X,y, test_size = 0.15, random_state = 99)
#Grid search!
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
cv = KFold(3, random_state = 1)
param_grid = {'n_estimators':[120,125,130,140],
              'max_depth':[10,30,35,40,50],
             'criterion':['gini', 'entropy']}
clf = GridSearchCV(ExtraTreesClassifier(random_state=1), 
                   param_grid = param_grid, scoring='accuracy', 
                   cv=cv).fit(X_train, y_train)
clf.best_estimator_, clf.best_score_
#Still overtrains.
clf = ExtraTreesClassifier(random_state = 1,
                          criterion = 'entropy',
                          max_depth = 10,
                          n_estimators = 120)
clf.fit(X_train, y_train)
print("Extra trees:", clf.score(X_train, y_train),
      " Accuracy:", clf.score(X_test, y_test))
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
pred = clf.predict(X_test)
confusion_matrix(y_test, pred)
print("ExtraTrees Accuracy:",accuracy_score(y_test, pred))
print("ExtraTrees Precision:",precision_score(y_test, pred))
print("ExtraTrees Recall:",recall_score(y_test, pred))
#Random forest
from sklearn.ensemble import RandomForestClassifier
cv = KFold(3, random_state = 1)
param_grid = {'n_estimators':[95,105,115],
              'max_depth':[10,30,50,100,200],
             'criterion':['gini', 'entropy']}
clf = GridSearchCV(RandomForestClassifier(random_state=1), 
                   param_grid = param_grid, scoring='accuracy', 
                   cv=cv).fit(X_train, y_train)
clf.best_estimator_, clf.best_score_
#This one also overtrains
clf = RandomForestClassifier(random_state = 1,
                          criterion = 'entropy',
                          max_depth = 10,
                          n_estimators = 115)
clf.fit(X_train, y_train)
print("Random Forest:", clf.score(X_train, y_train),
      " Accuracy:", clf.score(X_test, y_test))
#confusion matrix
pred = clf.predict(X_test)
confusion_matrix(pred, y_test)
print("RandomForest Accuracy:",accuracy_score(y_test, pred))
print("RandomForest Precision:",precision_score(y_test, pred))
print("RandomForest Recall:",recall_score(y_test, pred))
from sklearn.ensemble import AdaBoostClassifier
cv = KFold(3, random_state = 1)
param_grid = {'n_estimators':[500, 550, 600, 650],
              
             'learning_rate':[1.5, 1.6,1.7,1.8, 2],
             'algorithm': ['SAMME', 'SAMME.R']}
clf = GridSearchCV(AdaBoostClassifier(random_state=1), 
                   param_grid = param_grid, scoring='accuracy', 
                   cv=cv).fit(X_train, y_train)
clf.best_estimator_, clf.best_score_
#Better than 70%, and doesn't really overtrain.
clf = AdaBoostClassifier(learning_rate = 1.8,
                          algorithm = 'SAMME.R',
                          
                          n_estimators = 600)
clf.fit(X_train, y_train)
print("AdaBoost:", clf.score(X_train, y_train),
      " Accuracy:", clf.score(X_test, y_test))
#confusion matrix
#best result with a simple model using basic machine learning algorithms.
pred = clf.predict(X_test)
confusion_matrix(pred, y_test)
print("AdaBoost Accuracy:",accuracy_score(y_test, pred))
print("AdaBoost Precision:",precision_score(y_test, pred))
print("AdaBoost Recall:",recall_score(y_test, pred))
