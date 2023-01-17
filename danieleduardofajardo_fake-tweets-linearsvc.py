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
df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
## import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import nltk

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

import re

import pickle

from nltk.tokenize import word_tokenize

def missing(dfs):

    for c in dfs.columns:

        print(c)

        print(df[c].isnull().sum()/len(df))

def score(y_test, pred):

    print('F1 Score:', np.round(metrics.f1_score(y_test, pred,average='weighted'),4))

    print('acc Score:', np.round(metrics.accuracy_score(y_test, pred),4))



stops = np.array(stopwords.words('english'))

l = WordNetLemmatizer()

p = PorterStemmer()    

def stopping(variable):

    status=True

    if (variable in stops): 

        status = False

    return status

def s(ft):

    ft = re.sub('[^a-zA-Z]', ' ', ft)

    lower = ft.lower()

    tokens = word_tokenize(lower)

    filtered = filter(stopping, tokens)

    tok=[]

    for s in filtered: 

        tok.append(s)

        tokens = [p.stem(token) for token in tok]

        tokens = [l.lemmatize(token) for token in tokens]

    return ' '.join(tokens)



def preprocess(df):

    for c in df.columns:

        df.loc[df[c].isnull() ,c]=" "

    df["text"]  = df["text"] .apply(s)

    df["full"] =  df["keyword"]+" "+ df["location"]+" "+df["text"]

    x_words=df["full"]

    y=df["target"]

    y=y.astype('int')

    tv = TfidfVectorizer(min_df=0, max_df=0.5, ngram_range=(1,2),sublinear_tf=False)

    x= tv.fit_transform(x_words)

    return x, y, tv

x, y, _= preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=0,train_size =0.99)

lsvc = LinearSVC()

lsvc.fit(X_train, y_train)

pred = lsvc.predict(X_test)

score(y_test,pred)
lsvc = LinearSVC()

x, y, tv = preprocess(df)

lsvc.fit(x,y)





df1 = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

for c in df1.columns:

    df1.loc[df1[c].isnull() ,c]=" "

print("PREPROCESSING 2")

df1["text"]  = df1["text"] .apply(s)

df1["full"] =  df1["keyword"]+" "+ df1["location"]+" "+df1["text"]

x_words=df1["full"]



x= tv.transform(x_words)

print(x.shape)

pred = lsvc.predict(x)



len(pred)



df1["target"]=pred



df1[["id","target"]].to_csv("/kaggle/working/SVC.csv",index=False)