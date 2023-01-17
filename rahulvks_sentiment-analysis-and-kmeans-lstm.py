%matplotlib inline

import re

import sqlite3

import pandas as pd

import numpy as np

import nltk

import tqdm as tqdm

import string

from nltk.corpus import stopwords

stop = stopwords.words("english")

import matplotlib.pyplot as plt

import numpy as np

import datetime as dt

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer

english_stemmer=nltk.stem.SnowballStemmer('english')



from gensim import summarization

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from math import floor,ceil



from sklearn.svm import LinearSVC



from keras.models import Sequential

from keras.layers import LSTM, Dense, Embedding

import seaborn as sns

data = pd.read_csv('../input/Reviews.csv',nrows=20000,encoding='utf8')
data.columns
data=data.drop(['UserId','Id','ProfileName','Time'],axis=1)
sns.countplot(data['Score'])
def cleaning( review, remove_stopwords=True):

   

    



    review_text = re.sub("[^a-zA-Z]"," ", review)

   

    words = review_text.lower().split()

    

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]



    b=[]

    stemmer = english_stemmer 

    for word in words:

        b.append(stemmer.stem(word))



    

    return(b)
Test = "###$$$$$$$999$$$$*&^^saello the if area"

cleaning(Test)
clean_Text = []

for review in data['Text']:

    clean_Text.append( " ".join(cleaning(review)))

    

clean_summary = []

for review in data['Summary']:

    clean_summary.append( " ".join(cleaning(review)))
Top_Words_Review =pd.Series(' '.join(clean_Text).lower().split()).value_counts()[:10]

print ("Top Count Words Used In Review", Top_Words_Review)
Top_Words_Summary = pd.Series(' '.join(clean_summary).lower().split()).value_counts()[:10]

print ("Top Count Words Used In Summary",Top_Words_Summary)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=4, max_features = 10000)

vz = vectorizer.fit_transform(clean_Text)

tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

print("use: " + str(tfidf["br"]))

print("like: " + str(tfidf["like"]))

print("tast: " + str(tfidf["tast"]))

print("flavor: " + str(tfidf["flavor"]))

print("coffe: " + str(tfidf["coffe"]))
from nltk.sentiment.vader import SentimentIntensityAnalyzer

Senti = SentimentIntensityAnalyzer()

sample_review = clean_Text[:5]

for sentence in sample_review:

    sentence

    ss = Senti.polarity_scores(sentence)

    for k in sorted(ss):

        print('{0}: {1}, '.format(k, ss[k]))

    print(sentence) 
from sklearn.cluster import MiniBatchKMeans



num_clusters = 10

kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, 

                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000)

kmeans = kmeans_model.fit(vz)

kmeans_clusters = kmeans.predict(vz)

kmeans_distances = kmeans.transform(vz)

sorted_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(num_clusters):

    print("Cluster %d:" % i)

    for j in sorted_centroids[i, :5]:

        print(' %s' % terms[j])

    print()
Text = data.Text

Ratings = data.Score

vectorizer = TfidfVectorizer(max_df=.8)

vectorizer.fit(Text)

def categorize(ratings):

    cats = []

    for rating in ratings:

        v = [0,0,0,0,0]

        v[rating-1] = 1

        cats.append(v)

    return np.array(cats)



X = vectorizer.transform(Text).toarray()

y = categorize(Ratings.values)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)



model = Sequential()

model.add(Dense(128,input_dim=X_train.shape[1]))

model.add(Dense(5,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(X_train,y_train,nb_epoch=10,batch_size=32,verbose=1)

model.evaluate(X_test,y_test)[1]




