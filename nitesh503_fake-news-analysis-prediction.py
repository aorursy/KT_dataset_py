import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
Train = pd.read_csv("../input/fake-news-data/kaggle_fake_train.csv")

# here we are printing first five lines of our train dataset
Train.head()
# here we are Getting the Independent Features
X=Train.drop('label',axis=1)
# printing head of our independent features
X.head()
# here we are printing shape of our dataset
Train.shape

# here we are checking if there is null value or not
Train.isnull().sum()
# here we are droping NaN values from our dataset
Train=Train.dropna()
# here we are checking again if there is any NaN value or not
Train.isnull().sum()
Train.head(10)
import seaborn as sns
sns.catplot('label', data=Train, kind='count')

import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly
labels = ['0',' 1']
values = [
      len(Train[(Train["label"] == 0)]), 
      len(Train[(Train["label"] == 1)]), 
]
colors = ['#FEBFBB', '#E13966']

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))

plotly.offline.iplot([trace], filename='styled_pie_chart')
# here we are copying our dataset .
Train=Train.copy()
# here we are reseting our index
Train.reset_index(inplace=True)
# here we are printing our first 10 line of dataset for checking indexing
Train.head(10)
x=Train['title']
# here we are making independent features
y=Train['label']
y.shape
# here we are importing nltk,stopwords and porterstemmer we are using stemming on the text 
# we have and stopwords will help in removing the stopwords in the text

#re is regular expressions used for identifying only words in the text and ignoring anything else
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(Train)):
    review = re.sub('[^a-zA-Z]', ' ', Train['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
corpus[30]
# here we are setting vocabulary size
voc_size=5000
# here we are performing one hot representation
from tensorflow.keras.preprocessing.text import one_hot
one_hot_rep=[one_hot(words,voc_size)for words in corpus] 
# here we are printing length of first line
len(one_hot_rep[0])
# here we are printing length of 70 line
len(one_hot_rep[70])
# here we are importing library for doind padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
# here we are specifying a sentence length so that every sentence in the corpus will be of same length

sentence_length=25

# here we are using padding for creating equal length sentences


embedded_docs=pad_sequences(one_hot_rep,padding='pre',maxlen=sentence_length)
print(embedded_docs)
z =np.array(embedded_docs)
y =np.array(y)
# here we are printing shape 
z.shape,y.shape
# here we are splitting the data for training and testing the model

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(z, y, test_size=0.10, random_state=42)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, x_train, y_train, x_test, y_test, train=True):
    if train:
        pred = clf.predict(x_train)
        print("Train Result:\n===========================================")
        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")
        print(f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, clf.predict(x_train))}\n")
        
    elif train==False:
        pred = clf.predict(x_test)
        print("Test Result:\n===========================================")        
        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")
        print(f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
from sklearn.ensemble import BaggingClassifier
bagging_clf = BaggingClassifier(n_estimators=10, random_state=42)
bagging_clf.fit(x_train, y_train)
print_score(bagging_clf, x_train, y_train, x_test, y_test, train=True)
print_score(bagging_clf, x_train, y_train, x_test, y_test, train=False)
from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier(random_state=42, n_estimators=1000)
rand_forest.fit(x_train, y_train)
print_score(rand_forest, x_train, y_train, x_test, y_test, train=True)
print_score(rand_forest, x_train, y_train, x_test, y_test, train=False)
from sklearn.ensemble import AdaBoostClassifier

ada_boost_clf = AdaBoostClassifier(n_estimators=30)
ada_boost_clf.fit(x_train, y_train)
print_score(ada_boost_clf, x_train, y_train, x_test, y_test, train=True)
print_score(ada_boost_clf, x_train, y_train, x_test, y_test, train=False)
from sklearn.ensemble import GradientBoostingClassifier

grad_boost_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
grad_boost_clf.fit(x_train, y_train)
print_score(grad_boost_clf, x_train, y_train, x_test, y_test, train=True)
print_score(grad_boost_clf, x_train, y_train, x_test, y_test, train=False)