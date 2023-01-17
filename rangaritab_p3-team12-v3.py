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
train_df = pd.read_csv("../input/train.csv")

train_df = train_df[['id','comment_text', 'target']]

test_df = pd.read_csv("../input/test.csv")
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

def stemming(sentence):

    stemSentence = ""

    for word in sentence.split():

        stem = stemmer.stem(word)

        stemSentence += stem

        stemSentence += " "

    stemSentence = stemSentence.strip()

    return stemSentence

  

train_df['comment_text'] = train_df['comment_text'].apply(stemming)
from sklearn.feature_extraction.text import TfidfVectorizer

import re, string

re_tok = re.compile(f'([{string.punctuation}])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()



vect = TfidfVectorizer(input="content", 

                encoding="utf-8", 

                decode_error="strict", 

                strip_accents="unicode", 

                lowercase=True, 

                preprocessor=None, 

                tokenizer=tokenize, 

                analyzer="word", 

                stop_words=None, 

                token_pattern="(?u)\b\w\w+\b", 

                ngram_range=(1, 2), 

                max_df=0.9, 

                min_df=3, 

                max_features=None, 

                vocabulary=None, 

                binary=False, 

                norm="l2", 

                use_idf=1, 

                smooth_idf=1, 

                sublinear_tf=1)
X = vect.fit_transform(train_df["comment_text"])

y = np.where(train_df['target'] >= 0.5, 1, 0)



test_X = vect.transform(test_df["comment_text"])
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier  



from sklearn.model_selection import train_test_split, cross_val_score



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)
%%time

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(penalty="l2", 

                                             dual=False, 

                                             tol=0.0001, 

                                             C=1.0, 

                                             fit_intercept=True, 

                                             intercept_scaling=1, 

                                             class_weight="balanced", 

                                             random_state=None, 

                                             solver="liblinear", 

                                             max_iter=100, 

                                             multi_class="auto", 

                                             verbose=0, 

                                             warm_start=False, 

                                             n_jobs=None)



lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
cv_accuracy = cross_val_score(

    LogisticRegression(C=5, random_state=42, solver='sag', max_iter=1000, n_jobs=-1), 

    X, y, cv=5, scoring='roc_auc'

)

print(cv_accuracy)

print(cv_accuracy.mean())
accuracy_score(y_test, y_pred)
prediction = lr.predict_proba(test_X)[:,1]
submission = pd.read_csv("../input/sample_submission.csv")

submission['prediction'] = prediction

submission.to_csv('submission.csv', index=False)