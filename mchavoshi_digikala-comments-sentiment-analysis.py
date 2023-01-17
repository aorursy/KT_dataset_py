# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from spacy.lang.fa import stop_words
from string import punctuation, printable, digits
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/digikala-comments-persian-sentiment-analysis/data.csv')
df.head()
df.shape
df.describe()
df.sort_values(by='Score').head()
# column 'Suggestion' is not needed
df = df.drop(['Suggestion'], axis=1)
df.head()
df['Score'] = (df['Score']/10).astype('int')
# filter only extreme cases, cases in betweem are a combination of pos and neg reviews
df = df.loc[(df['Score']<6) | (df['Score']>7)]
df.loc[:,'label'] = df['Score'].apply(lambda score: 1 if score>7 else 0)
plt.bar(['Positive','Negative'], df['label'].value_counts()/df.shape[0])
train_X, test_X, train_y, test_y = train_test_split(df['Text'].values, df['label'].values, stratify=df['label'])
train_X.shape, train_y.shape, test_X.shape, test_y.shape
def load_stopwords():
    f = open("/kaggle/input/farsi-stopwords/fa_stop_words.txt", "r", encoding='utf8')
    stopwords = f.read()
    stopwords = stopwords.split('\n')
    stopwords = set(stopwords)
    custom_stop_words = {'آنكه','آيا','بدين','براين','بنابر','میشه','میکنه','باشه','سلام','میکشه','اونی',''}
    stopwords = stopwords | stop_words.STOP_WORDS | custom_stop_words
    # excluding space
    stopwords = list(stopwords)[1:]
    unwanted_num = {'خوش','بهتر','بد','خوب','نیستم','عالی','نیست','فوق','بهترین'} 
    stopwords = [ele for ele in stopwords if ele not in unwanted_num] 
    return stopwords
# we make a transformer so that we can use it in a pipeline
class Preprocess(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words):
        self.stop_words = stop_words
    def fit(self, X, y=None):
        return self
    def transform(self, corpus):
        res = []
        for data in corpus:
            if not self.stop_words:
                self.stop_words = set([])
            ## ensure working with string
            doc = str(data)
            # First remove punctuation form string
            PUNCT_DICT = {ord(punc): None for punc in punctuation+'،'}
            doc = doc.translate(PUNCT_DICT)
            # remove numbers
            doc = doc.translate({ord(k): None for k in digits})
            tokens = doc.split()
            tokens = [t for t in tokens if len(t) > 1]
            res.append(' '.join(w for w in tokens if w not in self.stop_words))
        return res
idx = 10
print(train_X[idx])
Preprocess(load_stopwords()).transform([train_X[idx]])
text_clf = Pipeline([
    ('prep', Preprocess(load_stopwords())),
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(train_X, train_y)
print(classification_report(train_y, text_clf.predict(train_X)))
test_pred = text_clf.predict(test_X)
print(classification_report(test_y, test_pred))
np.array(test_X)[test_pred!=test_y][10:20]
