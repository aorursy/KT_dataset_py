# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np
train_path = "/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv"

train_data = pd.read_csv(train_path)
train_data.head(3)
train_data = train_data.drop('id',axis=1)
size = train_data.shape[0]

print(size)
import seaborn as sns

sns.countplot(train_data['label'])
import nltk

import re

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
def prepare_corpus(tweets):

  corpus_tweets = []

  size = tweets.shape[0]

  ps = PorterStemmer()

  for i in range(0,size):

    tweet = re.sub(pattern='[^a-zA-Z]',repl=' ', string=tweets['tweet'][i])



    tweet = re.sub(pattern='user' , repl='' , string = tweet)



    tweet = tweet.lower()



    words = tweet.split()



    words = [ps.stem(word) for word in words if not word in stopwords.words('english')]



    tweet = ' '.join(words)



    corpus_tweets.append(tweet)

  return corpus_tweets



corpus_tweets_train = prepare_corpus(train_data)
corpus_tweets_train[0:2]
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=7000)

X_tfidf = tfidf.fit_transform(corpus_tweets_train).toarray()

y_ifidf = train_data['label'].values
X_tfidf[0:2]
from sklearn.model_selection import train_test_split

def split_train_test(X,y):

  X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.20)

  return X_train , X_test , y_train , y_test



X_train_idf , X_test_idf , y_train_idf , y_test_idf = split_train_test(X_tfidf, y_ifidf)
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

def accuracy_check(model,data,label):

  y_pred = model.predict(data)

  print(classification_report(label , y_pred)) 

  accuracy = accuracy_score(label , y_pred)

  return accuracy
from sklearn.naive_bayes import MultinomialNB

nb_idf = MultinomialNB()

nb_idf.fit(X_train_idf , y_train_idf)

nb_idf_accuracy = accuracy_check(nb_idf , X_test_idf , y_test_idf)

print(nb_idf_accuracy)
def optimization_idf(X_train_idf , X_test_idf , y_train_idf , y_test_idf):

  best_accuracy = 0.0

  alpha_val = 0.0

  for i in np.arange(0.1,1.1,0.1):

    temp_classifier = MultinomialNB(alpha=i)

    temp_classifier.fit(X_train_idf, y_train_idf)

    temp_y_pred = temp_classifier.predict(X_test_idf)

    score = accuracy_score(y_test_idf, temp_y_pred)

    print("Accuracy score for alpha={} is: {}%".format(round(i,1), round(score*100,2)))

    if score>best_accuracy:

      best_accuracy = score

      alpha_val = i

  print('The best accuracy is {}% with alpha value as {}'.format(round(best_accuracy*100, 2), round(alpha_val,1)))

  return alpha_val



optimal_value_idf = optimization_idf(X_train_idf , X_test_idf , y_train_idf , y_test_idf)
ml_model_final = MultinomialNB(alpha = 0.1)

ml_model_final.fit(X_tfidf , y_ifidf)
test_path = "/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv"

test_data = pd.read_csv(test_path)
test_data.head(3)
corpus_test = prepare_corpus(test_data)

vectors = tfidf.transform(corpus_test).toarray()
answer = ml_model_final.predict(vectors)
submission = test_data

submission.head(3)
submission['Predicted Labels'] = answer
submission.head()
ones = [ans for ans in answer if ans==1]

len(ones)
import seaborn as sns

sns.countplot(submission['Predicted Labels'])
submission.to_csv('submission.csv' , index=False)