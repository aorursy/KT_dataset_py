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
import nltk

import pandas as pd

import re

from sklearn.feature_extraction.text import TfidfVectorizer

import string
data = pd.read_csv("/kaggle/input/SMSSpamCollection.tsv", sep='\t')

data.columns = ['label', 'body_text'] # Giving the column columns
data.head()
data.isnull().sum()
data.label.value_counts()
data['text_cleaned']=data['body_text'].apply(lambda x: ''.join(word for word in x if word not in string.punctuation))
data.head() # In in index 2,4,5 we can see the changes occured
stopwords = nltk.corpus.stopwords.words('english')
data['text_cleaned']=data['text_cleaned'].apply(lambda x: ' '.join(word for word in re.split('\W+', x) if word.lower() not in stopwords))

data.head()
wn = nltk.WordNetLemmatizer()
data['text_cleaned']=data['text_cleaned'].apply(lambda x: ' '.join(wn.lemmatize(word) for word in re.split('\W+', x)))
data.head()
from sklearn.feature_extraction.text import CountVectorizer



count_vect = CountVectorizer()

X_counts = count_vect.fit_transform(data['text_cleaned'])

print(X_counts.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()

X_tfidf = tfidf_vect.fit_transform(data['text_cleaned'])

print(X_tfidf.shape)
def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")), 3)*100
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))

data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))
data.head()
X_features = pd.concat([data['body_len'], data['punct%'], pd.DataFrame(X_tfidf.toarray())], axis=1)

X_features.head()
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, data['label'], test_size=0.2)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)

rf_model = rf.fit(X_train, y_train)
sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]
y_pred = rf_model.predict(X_test)

precision, recall, fscore, support = score(y_test, y_pred, pos_label='spam', average='binary')
print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),

                                                        round(recall, 3),

                                                        round((y_pred==y_test).sum() / len(y_pred),3)))