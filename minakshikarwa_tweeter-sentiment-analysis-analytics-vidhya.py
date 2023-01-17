# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import re

from nltk.tokenize import word_tokenize

from string import punctuation 

from nltk.corpus import stopwords 

from nltk.stem import WordNetLemmatizer



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

test = pd.read_csv("../input/test_oJQbWVk.csv")

train = pd.read_csv("../input/train_2kmZucJ.csv")
lemma = WordNetLemmatizer()

stop = set(stopwords.words('english')+list('punctuation'))

len(stop)
# preprocessing of tweets

def processTweet(tweet):

        tweet = tweet.lower() # convert text to lower-case

        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs

        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames

        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag

        tweet = re.sub('[^a-z]',' ',tweet)

        tweet = re.sub(r'\b\w{1,2}\b', '', tweet) # remove words length upto 2 letters

        tweet = word_tokenize(tweet) # remove repeated characters (helloooooooo into hello)

        return [lemma.lemmatize(word) for word in tweet if word not in stop]
X = train['tweet']

Y = train['label']
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.20, random_state = 2)
tfidf_vectorizer = TfidfVectorizer(analyzer = processTweet, max_df = 0.95, min_df = 10, stop_words=stop)

# TF-IDF feature matrix

tfidf = tfidf_vectorizer.fit(x_train)

train1 = tfidf.transform(x_train)

val1 = tfidf.transform(x_val)
clf= MultinomialNB()

clf.fit(train1, y_train)
predictions=pd.DataFrame(list(zip(y_val,clf.predict(val1))),columns=['real','predicted'])

pd.crosstab(predictions['real'],predictions['predicted'])
print("Accuracy Score on Validation:", accuracy_score(clf.predict(val1),y_val)*100,"%")

print("F1 Score:", f1_score(clf.predict(val1),y_val)*100,"%")
test_x = test['tweet']

test1 = tfidf.transform(test_x)
label = clf.predict(test1)

label