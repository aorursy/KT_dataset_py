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
import unicodedata, re
df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
X_train , y_train = df["text"], df["target"]

X_test = df_test["text"]
df_test.head()
def preprocess(X):

    X = pd.Series(X).astype('str')

    X = X.apply(lambda val: unicodedata.normalize('NFKD', val).encode('ascii', 'ignore').decode())

    X = X.apply(lambda val: val.lower())

    X = X.apply(lambda val: re.sub(r"([?.!,¿])", r" \1 ", val))

    X = X.apply(lambda val: re.sub(r'[" "]+', " ", val))

    X = X.apply(lambda val: re.sub(r"[^a-zA-Z?.!,¿]+", " ", val))

    X = X.apply(lambda val: val.strip())

    return X



X_train,  X_test = [preprocess(x) for x in [X_train, X_test]]

X_train_pre = X_train
X_train_pre[99]
import nltk

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english')) 



def stop_lemmatize(X):

    X = X.apply(lambda x : word_tokenize(x))

    X = X.apply(lambda x : [k for k in x if k not in stop_words])

    X = X.apply(lambda x : [wordnet_lemmatizer.lemmatize(k) for k in x])

    X = X.apply(lambda x : " ".join(x))

    return X

X_train, X_test = [stop_lemmatize(x) for x in [X_train, X_test]]
X_train[99]
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

newX = list(X_train) + list(X_test)

X = vectorizer.fit_transform(newX)

X_train, X_test = X[:len(X_train)], X[len(X_train):]

y_train = np.asarray(y_train, dtype = np.float32)

from sklearn.model_selection import train_test_split

#X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, y_train, test_size=0.05)

#print([i.shape for i in [X_train, X_valid, Y_train, Y_valid]])
import lightgbm as lgbm



params = {

    'objective' :'binary',

    'learning_rate' : 0.02,

    'bagging_fraction': 0.8, 

    'bagging_freq':1,

    'boosting_type' : 'gbdt',

    'metric': 'binary_logloss'

}



d_train = lgbm.Dataset(X_train, y_train)

#d_valid = lgbm.Dataset(X_valid, Y_valid)

bst = lgbm.train(params, d_train, 5000)
pred = bst.predict(X_test)
holdout = pd.DataFrame({'id': df_test.id, 'target': pd.Series(pred).apply(lambda x : 1 if x > 0.5 else 0)})

#write the submission file to output

holdout.to_csv('submission.csv', index=False)