# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import roc_curve,auc

from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import confusion_matrix



# Any results you write to the current directory are saved as output.

# pd.read_json('../input/Sarcasm_Headlines_Dataset.json', lines = True)
def parseJson(fname):

    for line in open(fname,'r'):

        yield eval(line)
file_name = '../input/Sarcasm_Headlines_Dataset.json'

data = list(parseJson(file_name))
df = pd.DataFrame(data)

df.head()
df = df.drop('article_link', axis= 1)

df.head()
df['len'] = df['headline'].apply(lambda x: len(x.split(" ")))

df.head()
df_length = df['len'].value_counts().reset_index()

df_length.rename(columns={'index': 'length_word', 'len':'frequency'}, inplace = True)

df_length.head()
import matplotlib.pyplot as plt

plt.bar(df_length['length_word'], df_length['frequency'])
# removing those headline whose length is greather than 19

print('shape before preprocessing ',df.shape)

df = df[df['len'] < 19]

print('shape after preprocessing ',df.shape)

df.head()
sns.countplot(df['is_sarcastic'])
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features= 5000)

X = tf.fit_transform(df['headline'])

y = df['is_sarcastic']

print(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
nb = BernoulliNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

confusion_matrix(y_pred, y_test)
# Reciever opereating charactaristics (ROC)

proba = nb.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, proba)



auc_val = auc(fpr,tpr)

plt.plot(fpr, tpr)
f1_score(y_pred, y_test)
## xgboost classifier

from xgboost import XGBClassifier

for i in range(10, 12):

    model = XGBClassifier(max_depth = i, n_jobs=4 )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f1_score(y_pred, y_test))
# data cleaning





import string

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



wordnet_lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')

# print ("punctuations that gonna be removed, : ",string.punctuation)

# print(stop_words)



def process(x):

    token_list = x.split()

#     print(token_list)

    new_list = [w.lower() for w in token_list if w not in string.punctuation]

    new_list = [w for w in new_list if w not in stop_words]

    new_list = [wordnet_lemmatizer.lemmatize(w) for w in new_list]

    return " ".join(new_list)



df['headline'] = df['headline'].apply(process)
# XGB after cleaning

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features= 5000, token_pattern="[a-zA-Z]{2,}", norm='l1')

X = tf.fit_transform(df['headline'])

y = df['is_sarcastic']

df.drop('len', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)



for i in range(15, 20):

    model = XGBClassifier(max_depth = i, n_jobs = 8)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f1_score(y_pred, y_test))
tf.get_feature_names()[:10]
#naive bayes after cleaning

nb = BernoulliNB(alpha=1)

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

f1_score(y_pred, y_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

f1_score(y_pred, y_test)
vec = CountVectorizer(ngram_range=(1,2), max_features=5000,  token_pattern="[a-zA-Z]{2,}")

X = vec.fit_transform(df['headline'])

y = df['is_sarcastic']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)
nb = BernoulliNB(alpha=1)

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

f1_score(y_pred, y_test)
from sklearn.linear_model import SGDClassifier
model = SGDClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1_score(y_pred, y_test)
from sklearn.svm import LinearSVC
model = LinearSVC()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1_score(y_pred, y_test)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1_score(y_pred, y_test)