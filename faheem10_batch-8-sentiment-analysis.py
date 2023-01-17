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
Dataset = pd.read_csv('../input/Tweets.csv')
Dataset.head()
Dataset.text[30]
Dataset = Dataset[Dataset['airline_sentiment_confidence'] > 0.60]
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn import metrics
Dataset = Dataset[['text', 'airline_sentiment']]
Dataset.head()
Dataset.columns = ['text', 'label']
Dataset.head()
le = LabelEncoder()
Dataset['label'] = le.fit_transform(Dataset['label'])
Dataset.head()
print(le.classes_)
def tokenize(sen):
    tknzr = TweetTokenizer()
    return tknzr.tokenize(sen)


en_stopwords = stopwords.words('english')
vectorizer = CountVectorizer(analyzer='word',  
                             ngram_range=(1,1),
                             tokenizer = tokenize,
                             lowercase= True, 
                             stop_words=en_stopwords)

pipe = make_pipeline(vectorizer, SVC(gamma = 'auto'))

kf = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = True)
X_train, X_test, y_train, y_test = train_test_split(Dataset['text'].values, Dataset['label'].values, test_size = 0.2)
model = GridSearchCV(estimator=pipe,    
                     cv=kf, 
                     param_grid={'svc__C': [0.1, 1]},
                    scoring = 'accuracy')
model.fit(X_train, y_train)

model.score(X_test, y_test)
model.best_score_
model.best_estimator_
model.best_params_
X_test[40]
le.inverse_transform(model.best_estimator_.predict([X_test[40]]))[0]