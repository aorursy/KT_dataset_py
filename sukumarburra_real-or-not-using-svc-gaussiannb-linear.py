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
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
import numpy as np 

import pandas as pd 

import os



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from sklearn import linear_model

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
X = train["text"]

y = train["target"]

X_test = test["text"]

X.shape, y.shape, X_test.shape
X_for_tf_idf = pd.concat([X, X_test])

#tfidf = TfidfVectorizer()

tfidf = TfidfVectorizer(stop_words = 'english')

tfidf.fit(X_for_tf_idf)

X = tfidf.transform(X)

X_test = tfidf.transform(X_test)

del X_for_tf_idf
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
parameters = { 

    'gamma': [0.7, 1, 'auto', 'scale']

}

svc_model = GridSearchCV(SVC(kernel='rbf'), parameters, cv=4, n_jobs=-1).fit(X_train, y_train)



y_val_pred = svc_model.predict(X_val)

print (accuracy_score(y_val, y_val_pred), f1_score(y_val, y_val_pred))



y_test_pred = svc_model.predict(X_test)

y_test_pred



sub_df["target"] = y_test_pred

sub_df.to_csv("submission_svc.csv",index=False)
nb_model = GaussianNB()

nb_model.fit(X_train.todense(), y_train)

y_val_pred = nb_model.predict(X_val.todense())

print ( accuracy_score(y_val, y_val_pred), f1_score(y_val, y_val_pred) )



y_test_pred = nb_model.predict(X_test.todense())

y_test_pred



sub_df["target"] = y_test_pred

sub_df.to_csv("submission_nb.csv",index=False)
lin_model = linear_model.LogisticRegression()

lin_model.fit(X_train, y_train)

y_val_pred = lin_model.predict(X_val)

print (accuracy_score(y_val, y_val_pred), f1_score(y_val, y_val_pred))



y_test_pred = lin_model.predict(X_test)

y_test_pred



sub_df["target"] = y_test_pred

sub_df.to_csv("submission_lin.csv",index=False)