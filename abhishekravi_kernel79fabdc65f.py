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
train = pd.read_csv('/kaggle/input/imdb-review/train_data.csv')
test = pd.read_csv('/kaggle/input/imdb-review/test_data.csv')
train.head()
test.head()
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()
def clean(text):
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text
train['SentimentText'] = train['SentimentText'].apply(lambda x: clean(x))
test['Sentiment'] = test['Sentiment'].apply(lambda x : clean(x))
test.head()
from sklearn.model_selection import train_test_split
X = train[['SentimentText']].values
y = train[['Sentiment']].values
X_val = test[['Sentiment']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
(X_train.ravel()).shape
(X_test)[0]
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return bag-of-words representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5, token_pattern='(\S+)')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train.ravel())
    X_val_tfidf = tfidf_vectorizer.transform(X_val.ravel())
    X_test_tfidf = tfidf_vectorizer.transform(X_test.ravel())
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer.vocabulary_
X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train_tfidf,y_train.ravel())
y_pred = classifier.predict(X_test_tfidf)
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)
y_predicted = classifier.predict(X_val_tfidf)
y = pd.DataFrame(y_predicted,columns = ['sentiment-value'])
test = pd.concat([test,y],axis = 1)
test.head()
