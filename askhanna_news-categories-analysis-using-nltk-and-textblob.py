# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import re, textblob

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import classification_report, confusion_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/uci-news-aggregator.csv')
df.head()
df.shape[0]
cont = pd.DataFrame(columns=['Category Count'],data=df.groupby('CATEGORY')['ID'].count().values,index=df.groupby('CATEGORY')['ID'].count().index)
cont['Percent'] = cont['Category Count'].apply(lambda x:  x/df.shape[0])
print(cont)
def normalize_text(s):
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s

df['TEXT'] = [normalize_text(s) for s in df['TITLE']]
df.shape
countvector = CountVectorizer()
X_counts = countvector.fit_transform(df['TEXT'])
X_tf = TfidfTransformer(use_idf=False).fit_transform(X_counts)
y= df['CATEGORY']
y.shape
X_tf.shape
X_tf_train,X_tf_test,y_train,y_test = train_test_split(X_tf,y)
X_tf_train.shape
clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)
clf.fit(X_tf_train,y_train)
predicted = clf.predict(X_tf_test)
np.mean(predicted == y_test) 
print(list(np.unique(df['CATEGORY'])))
print(classification_report(y_pred = predicted,y_true=y_test,target_names=np.unique(df['CATEGORY'])))
clf_multiNB = MultinomialNB()
clf_multiNB.fit(X_tf_train,y_train)
predicted = clf_multiNB.predict(X_tf_test)
np.mean(predicted == y_test) 
X= df['TEXT']
y= df['CATEGORY']
X_train_pipe,X_test_pipe,y_train,y_test = train_test_split(X,y)
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',MultinomialNB())])
text_clf.fit(X_train_pipe,y_train)
predicted = text_clf.predict(X_test_pipe)
np.mean(predicted==y_test)

#X_tf_train,X_tf_test
params = {'vect__ngram_range':[(1,1),(1,2)],
          'tfidf__use_idf':(True,False),
          'clf__alpha':(1e-2,1e-3)}
gs_clf = GridSearchCV(text_clf,cv=5,param_grid=params,iid=False,n_jobs=-1)
#gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)

gs_clf.fit(X_train_pipe[:5000],y_train[:5000])
gs_clf.best_score_
for i in sorted(params.keys()):
    print('%s: %r ' %(i,gs_clf.best_params_[i]))