# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import string

%matplotlib inline
dataset_train = pd.read_csv('../input/twitter-sentiment-analysis-for-hate-speech/Twitter Sentiment Train.csv', sep = ',' )
#message for previous data cleaning method
#review = review.split()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#Stemming across data
ps = PorterStemmer()

dataset_train.head()
#Creating Corpus
corpus = []
for i in range (0,31962) :
    review = re.sub('[^a-zA-Z]', ' ', dataset_train['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    twitter_handle = ['user']
    review = [word for word in review if not word in twitter_handle]
    review = '  '.join(review)
    corpus.append(review)
dataset_train['corpus'] = corpus

dataset_train['corpus']= dataset_train['corpus'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
all_words = ' '.join([ word for word in corpus])
from wordcloud import WordCloud
wordcloud = WordCloud( width =800, height =500, random_state = 121, max_font_size = 110).generate(all_words)

plt.figure(figsize = (10,7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

positive_words = ' '.join([text for text in dataset_train['corpus'][dataset_train['label'] == 0]])
wordcloud = WordCloud( width =800, height =500, random_state = 121, max_font_size = 110).generate(positive_words)

plt.figure(figsize = (10,7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

negative_words = ' '.join([text for text in dataset_train['corpus'][dataset_train['label'] == 1]])
wordcloud = WordCloud( width =800, height =500, random_state = 121, max_font_size = 110).generate(negative_words)

plt.figure(figsize = (10,7))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(dataset_train['corpus']).toarray()
y = dataset_train.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from xgboost import XGBClassifier
model = XGBClassifier(eta = 0.3, max_depth = 5)
model.fit(X_train, y_train)
y_pred_XGB = model.predict_proba(X_test) # predicting on the validation set
y_pred_XGB = y_pred_XGB[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
y_pred_XGB = y_pred_XGB.astype(np.int)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred_XGB) # calculating f1 score
#Trying tf-idf 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df = 0.9, min_df = 2, max_features = 1500, stop_words = 'english')

X_tfidf = tfidf_vectorizer.fit_transform(dataset_train['corpus'])
y_tfidf = dataset_train.iloc[:, 1].values
from sklearn.cross_validation import train_test_split
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y_tfidf, test_size = 0.20, random_state = 0)
model_tfidf = XGBClassifier(eta = 0.3, max_depth = 10)
model_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_XGB_tfidf = model_tfidf.predict_proba(X_test_tfidf) # predicting on the validation set
y_pred_XGB_tfidf = y_pred_XGB_tfidf[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
y_pred_XGB_tfidf = y_pred_XGB_tfidf.astype(np.int)
from sklearn.metrics import f1_score
f1_score(y_test_tfidf, y_pred_XGB_tfidf) # calculating f1 score
