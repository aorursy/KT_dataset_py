

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import nltk.classify.util

# Creating the model and setting values for the various parameters

num_features = 300  # Word vector dimensionality

min_word_count = 40 # Minimum word count

num_workers = 4     # Number of parallel threads

context = 10        # Context window size

downsampling = 1e-3 # (0.001) Downsample setting for frequent words





#import pandas as pd

#import numpy as np

#import os

#import matplotlib.pyplot as plt

#import matplotlib

#matplotlib.style.use('ggplot')

#Cluster_Data = pd.read_csv('../input/7817_1.csv',nrows=6000)





import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import nltk

#nltk.download()

#import nltk.classify.util

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.classify import NaiveBayesClassifier

import numpy as np

import re

import string

import nltk

%matplotlib inline

Cluster_Data = pd.read_csv('../input/7817_1.csv',nrows=6000)

Cluster_Data.head()
Cluster_Data.isnull().sum()
Cluster_Data.shape
np.unique(Cluster_Data['brand'])
np.unique(Cluster_Data['categories'])
#Cluster_Data['dateAdded'].plot.hist(bins=10,figsize=(10,8));

import nltk

from nltk.corpus import stopwords

print(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer

import re

Cluster_Data.columns

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

new_stopwords = set(stopwords.words('english')) - {"not", "don't","didn't","doesn't"}

punctuations="?:!.,;"

from nltk.corpus import stopwords

def remove_stopword(word):

    return word not in words



from nltk.stem import WordNetLemmatizer

Lemma = WordNetLemmatizer()



from nltk.stem.snowball import SnowballStemmer

#stemmer = SnowballStemmer("english")

from nltk.stem import PorterStemmer

import re

regex=re.compile('[%s]' % re.escape(string.punctuation))

stemmer=PorterStemmer()



Cluster_Data['NewReviews'] = Cluster_Data['reviews.text'].str.lower().str.split()

#print(Cluster_Data['NewReviews'])

tok_rev=[]

for x in Cluster_Data['NewReviews']:

    new_review=[]

    for token in x:

        new_token=regex.sub(u'',token)

        if not new_token==u'':

            new_review.append(new_token)

    tok_rev.append(new_review)

#print(tok_rev)  

Cluster_Data['NewReviews'] = Cluster_Data['NewReviews'].apply(lambda x : [item for item in x if item not in new_stopwords])

#Cluster_Data['NewReviews'] = Cluster_Data['NewReviews'].apply(lambda x : [item for item in x if item not in ["."]])



#Cluster_Data['NewReviews'] = Cluster_Data["NewReviews"].apply(lambda x : [stemmer.stem(y) for y in x])

Cluster_Data['NewReviews'] = Cluster_Data["NewReviews"].apply(lambda x : [Lemma.lemmatize(y) for y in x])

print(Cluster_Data['NewReviews'])
Cluster_Data['Cleaned_reviews'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line))

for line in lists]).strip() for lists in Cluster_Data['NewReviews']] 

#Cluster_Data['Cleaned_reviews'] = WordNetLemmatizer().lemmatize(Cluster_Data['NewReviews'], pos="v")
Cluster_Data.shape
Cluster_Data['NewReviews']=[" ".join(review) for review in Cluster_Data['NewReviews'].values]
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = []

print("Parsing sentences from training set")

for review in Cluster_Data['NewReviews']:

    sentences += review

print(review)    

Newre=Cluster_Data['NewReviews']

print(Newre)

# Initializing the train model

num_features = 300  # Word vector dimensionality

min_word_count = 40 # Minimum word count

num_workers = 4     # Number of parallel threads

context = 10        # Context window size

downsampling = 1e-3 # (0.001) Downsample setting for frequent words



from gensim.models import word2vec

print("Training model....")

model = word2vec.Word2Vec(Newre,\

                          workers=num_workers,\

                          size=num_features,\

                          min_count=min_word_count,\

                          window=context,

                          sample=downsampling)



# To make the model memory efficient

model.init_sims(replace=True)



# Saving the model for later use. Can be loaded using Word2Vec.load()

model_name = "300features_40minwords_10context"

model.save(model_name)
#model.wv.doesnt_match("man woman dog child kitchen".split())
Cluster_Data.shape
Cluster_Data['NewRatings'] = Cluster_Data['reviews.rating']
Cluster_Data.shape
#Cluster_Data = Cluster_Data.dropna()

Cluster_Data=Cluster_Data.dropna(subset = ['NewRatings', 'NewReviews'])
Cluster_Data.shape
Cluster_Data.columns
Cluster_Data.head()
Cluster_Data.shape
Cluster_Data.columns
Reviews = Cluster_Data.NewReviews

len(Reviews)

Reviews = Reviews.dropna()

print(Reviews)
Ratings=Cluster_Data.NewRatings
#X_train = Cluster_Data.loc[:999, 'NewReviews'].values

#y_train = Cluster_Data.loc[:999, 'NewRatings'].values

#X_test = Cluster_Data.loc[1000:, 'NewReviews'].values

#y_test = Cluster_Data.loc[1000:, 'NewRatings'].values
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

clf = Pipeline([

    ('vect', CountVectorizer(stop_words= "english")),

    ('tfidf', TfidfTransformer()),

    ('classifier', MultinomialNB()),

    ])
print (Cluster_Data["NewReviews"])
Cluster_Data.shape


x = Cluster_Data.brand

y = Cluster_Data.NewReviews

xy=pd.concat([x,y],axis=1)

a=Cluster_Data[['NewReviews']]

#y=Cluster_Data.filter(['NewRatings'],axis=1)

columns = ['brand','NewReviews']

#df1 = pd.DataFrame(Cluster_Data, columns=columns)

df1 = pd.DataFrame(Cluster_Data, columns = ['brand', 'NewReviews']) 

print (xy)
x.shape,y.shape,df1.shape
print (df1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Cluster_Data["NewReviews"], Cluster_Data["NewRatings"], test_size = 0.25,random_state = 0)

#X_train, X_test, y_train, y_test = train_test_split(xy,Cluster_Data["NewRatings"] ,test_size = 0.25)

#X_train, X_test,y_train, y_test = train_test_split(Cluster_Data[["NewReviews","brand"]],Cluster_Data["NewRatings"],test_size=0.25,random_state = 0)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

#x_train.shape,y_train.shape
#X_train, X_test,y_train, y_test = train_test_split(x,y,test_size=0.25,random_state = 0)

#X_train.shape,X_test.shape,y_train.shape,y_test.shape
model = clf.fit(X_train,y_train)

#model = clf.fit(x_train,y_train)
print("Accuracy of Naive Bayes Classifier is {}".format(model.score(X_test,y_test)))