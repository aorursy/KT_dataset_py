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
# Load into Pandas dataframe
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
pd.set_option('max_colwidth', 400)
test[['id','text']].head()
# Make copies, drop id column
X_train = train.drop(['id','target'],axis=1)
y_train = train.target
X_test = test.drop(['id'],axis=1)
import re
# Check if hyperlinks exist

X_train['hyperlink'] = X_train['text'].str.contains('http')
X_test['hyperlink'] = X_test['text'].str.contains('http')

# Extract number of hashtags and mentions

def hash_num(text):
    text = len(re.findall(r"#(\w+)",text))
    return text

def ment_num(text):
    text = len(re.findall(r"@(\w+)",text))
    return text

X_train['num_hash'] = X_train['text'].apply(hash_num)
X_train['num_ment'] = X_train['text'].apply(ment_num)

X_test['num_hash'] = X_test['text'].apply(hash_num)
X_test['num_ment'] = X_test['text'].apply(ment_num)
# Clean up
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Remove stopwords

stop_words = stopwords.words('english')
def remove_stopwords(text):
    text = ' '.join([i for i in text.split(' ') if i not in stop_words ])
    return text

# Stemming

stemmer = SnowballStemmer("english")
def stemming(text):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

def clean_tweet(text):
    text = re.sub('http\S+','', text) # remove hyperlinks
    text = re.sub('@[A-Za-z0-9_]+', '', text) # remove mentions
    text = re.sub('[?|!|\'|"|#]','',text) # remove punctuation
    text = re.sub('[.|,|)|(|)|\|/]',' ',text) # remove punctuation
    text = re.sub('&amp; ','',text)
    text = re.sub('[0-9]*','', text) # remove numbers
    text = text.strip().lower()
    text = remove_stopwords(text)
    text = stemming(text)
    return text

X_train['text'] = X_train['text'].apply(clean_tweet)
X_test['text'] = X_test['text'].apply(clean_tweet)
X_train = X_train.drop(['keyword','location'],axis=1)
X_test = X_test.drop(['keyword','location'],axis=1)
X_train
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()

bow_train = count.fit_transform(X_train.text)
feature_train = count.get_feature_names()

bow_test = count.fit_transform(X_test.text)
feature_test = count.get_feature_names()
X_train = pd.concat([pd.DataFrame(bow_train.toarray(), columns=feature_train),
                     pd.get_dummies(X_train.hyperlink,prefix='hyper'),
                    X_train.num_hash,
                    X_train.num_ment], axis=1)
X_test = pd.concat([pd.DataFrame(bow_test.toarray(), columns=feature_test),
                     pd.get_dummies(X_test.hyperlink,prefix='hyper'),
                    X_test.num_hash,
                    X_test.num_ment], axis=1)
[X_train.shape,X_test.shape]
for col in X_train.columns.difference(X_test.columns):
    X_test[col]=0
for col in X_test.columns.difference(X_train.columns):
    X_train[col]=0
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=1)
nb.fit(X_train, y_train)
y_pred = pd.DataFrame(nb.predict(X_test),columns=['target'])
# Export predictions to csv
res = pd.concat([test.id,y_pred.target],axis=1)
res.to_csv(r'\kaggle\working\submission.csv', index = False)
res