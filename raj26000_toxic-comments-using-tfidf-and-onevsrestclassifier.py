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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
data_train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
print(data_train)
data_test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

test_id = data_test['id']
y_train = data_train.iloc[:,2:8]
y_train
StopWords = set(stopwords.words('english'))

def text_preprocess(text):
    trans = str.maketrans('','',string.punctuation)
    text = text.translate(trans)
    text = ' '.join([word.lower() for word in text.split() if word.lower() not in StopWords])
    return text

data_train['comment_text'] = data_train['comment_text'].apply(text_preprocess)
data_test['comment_text'] = data_test['comment_text'].apply(text_preprocess)
X_train = data_train['comment_text']
X_test = data_test['comment_text']
print(X_test.head())
X_train.head()
X_train = X_train.tolist()
X_test = X_test.tolist()

def lemmatize(data):
    lemmatizer = WordNetLemmatizer()
    data_lemm = []
    for text in data:
        lem_text = ''
        for word in text.split():
            lem_word = lemmatizer.lemmatize(word)
            lem_word = lemmatizer.lemmatize(lem_word, pos='v')
            lem_text = lem_text + ' ' + lem_word
        data_lemm.append(lem_text)
    return data_lemm
X_train_lemm = lemmatize(X_train)
X_test_lemm = lemmatize(X_test)
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9)
X_train_tfidf = tfidf.fit_transform(X_train_lemm)
X_test_tfidf = tfidf.transform(X_test_lemm)
clf = OneVsRestClassifier(LogisticRegression(penalty='l2',C=1)).fit(X_train_tfidf, y_train)
clf.predict(X_test_tfidf)
y_pred = clf.predict_proba(X_test_tfidf)
print(y_pred)
y_pred[:,0]
output_df = pd.DataFrame()
output_df['id'] = test_id
output_df['toxic'] = y_pred[:,0]
output_df['severe_toxic'] = y_pred[:,1]
output_df['obscene'] = y_pred[:,2]
output_df['threat'] = y_pred[:,3]
output_df['insult'] = y_pred[:,4]
output_df['identity_hate'] = y_pred[:,5]
print(output_df)
output_df.to_csv('Submission.csv', index=False)

