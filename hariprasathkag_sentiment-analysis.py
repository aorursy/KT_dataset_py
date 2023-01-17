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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
senti = SentimentIntensityAnalyzer()
senti.polarity_scores('i love india')
senti.polarity_scores('i LOVE india')
senti.polarity_scores('i LOVE!!! india')
senti.polarity_scores('i LOVE india :) ')
hotstar = pd.read_csv('../input/hotstar.allreviews_Sentiments.csv')
hotstar.head()
def get_sentiment(review):
    senti=SentimentIntensityAnalyzer()
    score=senti.polarity_scores(review)['compound']
    if score>0.05:
        sentiment='Positive'
    elif score < -0.05:
        sentiment='Negative'
    else:
        sentiment='Neutral'
    return sentiment
hotstar['sentiment_vader']=hotstar['Reviews'].apply(get_sentiment)
hotstar['sentiment_vader'].value_counts()
%matplotlib inline
hotstar['sentiment_vader'].value_counts().plot.barh(color='steelblue')
import nltk
from sklearn.feature_extraction.text import CountVectorizer
docs=hotstar['Reviews'].fillna('').str.lower()
docs=docs.str.replace('[^a-z ]','')
stopwords=nltk.corpus.stopwords.words('english')
stemmer=nltk.stem.PorterStemmer()
def clean_review(text):
    words=text.split(' ')
    word_root=[stemmer.stem(word) for word in words if word not in stopwords]
    return ' '.join(word_root)

docs_clean=docs.apply(clean_review)

vectorizer=CountVectorizer()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
train , test = train_test_split(docs_clean,test_size = 0.3 , random_state = 100)
vectorizer.fit(train)
train_dtm = vectorizer.transform(train)
test_dtm = vectorizer.transform(test)
train_x = train_dtm
train_y = hotstar.loc[train.index]['Sentiment_Manual']
test_x = test_dtm
test_y = hotstar.loc[test.index]['Sentiment_Manual']
model_nb = MultinomialNB()
model_nb.fit(train_x,train_y)
pred_class = model_nb.predict(test_x)
from sklearn.metrics import accuracy_score
print(accuracy_score(pred_class,test_y))
model_rf = RandomForestClassifier(n_estimators=300)
model_rf.fit(train_x,train_y)
pred_class = model_rf.predict(test_x)
print(accuracy_score(test_y,pred_class))
