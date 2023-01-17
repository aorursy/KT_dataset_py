import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))
import re
import logging
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

from bs4 import BeautifulSoup 
# удобная библиотека для обработки html-тегов, которые есть в текстах к этой задаче

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from gensim.models import Word2Vec 
# библиотека gensim, в которой реализовано много Deep Learning алгоритмов
# в том числе есть много алгортмов для обработки текста, в том числе тематическое моделирование

import nltk
# nltk.download()  # важно скачать датасеты, в том числе стоп-слова
from nltk.corpus import stopwords # сразу забираем стоп-слова

import warnings
warnings.filterwarnings('ignore')
%matplotlib notebook

import matplotlib.pyplot as plt
nltk.download('stopwords')
#set(stopwords.words('english'))
train = pd.read_csv("../input/labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)
test = pd.read_csv("../input/testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )
top_df = pd.DataFrame(top,columns=['cnt','tag'])
top_df.plot()
%%time

from sklearn.model_selection import train_test_split

result = []

for max_word in range(2,16,2):

    #df_train, features_full = formatData(df_clear, fit_transform = True,max_features = max_word
    X_train, X_test, y_train, y_test = train_test_split(features,df_clear['sentiment'].values,test_size = 0.15)
    
    forest = RandomForestClassifier(max_depth=12,random_state=17,n_estimators=60) 
    forest.fit( X_train, y_train )
    print(max_word,'ok')
    result.append([max_word, metrics.accuracy_score(forest.predict(X_test),y_test)])
result
metrics.accuracy_score(forest.predict(X_test),y_test)
clearData = []
df=train
for i in df.index:
    #print(i)
    test_text = BeautifulSoup(df.review[i]).get_text()

    words = re.sub("[^a-zA-Z]", " ", test_text)
    words = words.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]
    clearData.append( " ".join( meaningful_words ))


df['ClearReview'] = clearData


vectorizer = CountVectorizer(
                analyzer='word'
                , tokenizer=None
                , preprocessor = None
                , stop_words = None
                , max_features = 5000
            )

futures = vectorizer.fit_transform(df['ClearReview'].values).toarray()

vocab = vectorizer.get_feature_names()
#print(vocab[1:100])

dist = np.sum(futures, axis=0)
top=[]
for count, tag in sorted([(count, tag) for tag, count in zip(vocab, dist)], reverse=True)[1:5000]:
    top.append([count, tag])

top_df = pd.DataFrame(top,columns=['cnt','tag'])
from time import time 
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(futures,df['sentiment'].values,test_size = 0.15)
    
forest = RandomForestClassifier(max_depth=12,random_state=17,n_estimators=60) 
forest.fit( X_train, y_train )
print(metrics.accuracy_score(forest.predict(X_test),y_test))
#print(max_word,'ok')
#result.append([max_word, metrics.accuracy_score(forest.predict(X_test),y_test)])

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    # Print and return results
    print( "Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, pos_label=1), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print( f1, acc)
    print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

clf_A = LogisticRegression(random_state = 42)
clf_B = SVC(random_state = 912, kernel='rbf')
clf_C = lgb.LGBMClassifier(seed = 82)

train_predict(clf_A, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_B, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print ('')
clearData = []
df=test
for i in df.index:
    #print(i)
    test_text = BeautifulSoup(df.review[i]).get_text()

    words = re.sub("[^a-zA-Z]", " ", test_text)
    words = words.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]
    clearData.append( " ".join( meaningful_words ))


df['ClearReview'] = clearData


vectorizer = CountVectorizer(
                analyzer='word'
                , tokenizer=None
                , preprocessor = None
                , stop_words = None
                , max_features = 5000
            )

futures = vectorizer.fit_transform(df['ClearReview'].values).toarray()

vocab = vectorizer.get_feature_names()
#print(vocab[1:100])

dist = np.sum(futures, axis=0)
top=[]
for count, tag in sorted([(count, tag) for tag, count in zip(vocab, dist)], reverse=True)[1:5000]:
    top.append([count, tag])

top_df = pd.DataFrame(top,columns=['cnt','tag'])
