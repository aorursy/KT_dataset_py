# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

sample_df=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

test_df.head()

X_train=train_df['text']

X_test=test_df['text']

Y_train=train_df['target']
from sklearn.feature_extraction.text import CountVectorizer

vect=CountVectorizer().fit(X_train)

X_train_vect=vect.transform(X_train)

X_test_vect=vect.transform(X_test)

from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()



from sklearn.model_selection import cross_val_score

scores=cross_val_score(clf,X_train_vect,Y_train,cv=3,scoring='f1')

scores



from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

scores=cross_val_score(lr,X_train_vect,Y_train,cv=3,scoring='f1')

scores
from sklearn.svm import SVC

svc=SVC()

scores=cross_val_score(lr,X_train_vect,Y_train,cv=3,scoring='f1')

scores
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

scores=cross_val_score(rfc,X_train_vect,Y_train,cv=3,scoring='f1')

scores

from sklearn.feature_extraction.text import TfidfVectorizer

vect=TfidfVectorizer(min_df=5).fit(X_train)

X_train_vect2=vect.transform(X_train)

X_test_vect2=vect.transform(X_test)

#MultinomialNB on TFIDF

scores=cross_val_score(clf,X_train_vect2,Y_train,cv=3,scoring='f1')

scores

#LogisticRegression on TFIDF

scores=cross_val_score(lr,X_train_vect2,Y_train,cv=3,scoring='f1')

scores

#SupportVectorMachine on TFIDF

scores=cross_val_score(svc,X_train_vect2,Y_train,cv=3,scoring='f1')

scores

#RandomForestClassifier on TFIDF

scores=cross_val_score(rfc,X_train_vect2,Y_train,cv=3,scoring='f1')

scores

clf.fit(X_train_vect2,Y_train)

sample_df['target']=clf.predict(X_test_vect2)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)