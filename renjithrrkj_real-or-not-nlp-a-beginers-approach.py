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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

import re

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import fbeta_score, make_scorer

train=pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
print("the training dataset has",train.shape[0],'rows',train.shape[1],"columns")
plt.style.use('seaborn')

p1=sns.countplot(x='target',data=train)

for p in p1.patches:

        p1.annotate('{:6.2f}%'.format(p.get_height()/len(train)*100), (p.get_x()+0.1, p.get_height()+50))

        

plt.gca().set_ylabel('samples')



def cleaned(text):

    text = re.sub(r"\n","",text)

    text = text.lower()

    text = re.sub(r"\d","",text)        #Remove digits

    text = re.sub(r'[^\x00-\x7f]',r' ',text) # remove non-ascii

    text = re.sub(r'[^\w\s]','',text) #Remove punctuation

    text = re.sub(r'http\S+|www.\S+', '', text) #Remove http

    return text
train['cleaned'] = train['text'].apply(lambda x : cleaned(x))

test['cleaned'] = test['text'].apply(lambda x : cleaned(x))

train.head()
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



X = train['cleaned'].to_numpy()

y = train['target'].to_numpy()

for train_index, test_index in sss.split(X, y):

    print("TRAIN:", train_index, "TEST:", test_index)

    #X_train = X.loc[train_index]

    X_train, X_test = X[train_index], X[test_index]



    y_train, y_test = y[train_index], y[test_index]

tweets_pipeline = Pipeline([('CVec', CountVectorizer(stop_words='english')),

                     ('Tfidf', TfidfTransformer())])



X_train_transformed = tweets_pipeline.fit_transform(X_train)

X_test_transformed = tweets_pipeline.transform(X_test)
SVC_clf=SVC()
SVC_clf.fit(X_train_transformed,y_train)
y_pred=SVC_clf.predict(X_test_transformed)
print('accuracy of SVC classifier {}'.format(accuracy_score(y_pred,y_test)))
accuracy_scoring=make_scorer(accuracy_score)

params1={'C':[0.001, 0.01, 0.1, 1, 10, 100],'kernel':['poly', 'rbf', 'sigmoid']}

clf_gsc=GridSearchCV(SVC_clf,param_grid=params1,n_jobs=-1,scoring=accuracy_scoring)

clf_gsc.fit(X_train_transformed,y_train)
print('best score for Grid_searchCV',clf_gsc.best_score_)
print('best params for Grid_searchCV',clf_gsc.best_params_)
clf_best=clf_gsc.best_estimator_
clf_best.fit(X_train_transformed,y_train)
y_pred_tuned=clf_best.predict(X_test_transformed)
print('scores for best estimator',accuracy_score(y_pred_tuned,y_test))

y_pred
test_clean=test['cleaned'].to_numpy()

test_transformed = tweets_pipeline.transform(test_clean)

y_pred_test=SVC_clf.predict(test_transformed)

y_pred_series=pd.Series(y_pred_test,name='target')

sub=pd.concat([test['id'],y_pred_series ], axis=1)

sub.to_csv('submission.csv',index=False)

sub