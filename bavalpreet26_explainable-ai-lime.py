# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from collections import Counter
from imblearn.over_sampling import SMOTE
import matplotlib                  # 2D Plotting Library
import matplotlib.pyplot as plt
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
df = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')

rslt_df = df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]
rslt_df2 = df[(df['toxic'] == 1) | (df['severe_toxic'] == 1) | (df['obscene'] == 1) | (df['threat'] == 1) | (df['insult'] == 1) | (df['identity_hate'] == 1)]
new1 = rslt_df[['id', 'comment_text', 'toxic']].iloc[:23891].copy() 
new2 = rslt_df2[['id', 'comment_text', 'toxic']].iloc[:946].copy()
new = pd.concat([new1, new2], ignore_index=True)
new.tail()
#test train split
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new["comment_text"], new['toxic'], test_size=0.33)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5)
X1 = vectorizer.fit_transform(X_train)
X_test1= vectorizer.transform(X_test)
class_names = ['nontoxic', 'toxic']
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(C=0.1, solver='sag')
scores = cross_val_score(clf2, X1,y_train, cv=5,scoring='f1_weighted')
y_p1 = clf2.fit(X1, y_train).predict(X_test1)
from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p1)
print('Accuracy: %f' % accuracy)
from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, clf2)
new["comment_text"][0]
print(c.predict_proba([new["comment_text"][0]]))
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)
X_test = X_test.tolist()
X_test[0]
type(y_test)
y_test = y_test.tolist()
y_test = np.array(y_test)
type(y_test)
idx = 0
exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])


exp.as_list()
print('Original prediction:', clf2.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['you']] = 0
tmp[0,vectorizer.vocabulary_['thanks']] = 0
print('Prediction removing some features:', clf2.predict_proba(tmp)[0,1])
print('Difference:', clf2.predict_proba(tmp)[0,1] - clf2.predict_proba(X_test1[idx])[0,1])
%matplotlib inline
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('/tmp/oi.html')
exp.show_in_notebook(text=True)
print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(random_state=12)
x_train_res, y_train_res = sm.fit_sample(X1, y_train)
print('Resampled dataset shape %s' % Counter(y_train_res))
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=0.1, solver='sag')
scores = cross_val_score(clf, x_train_res,y_train_res, cv=5,scoring='f1_weighted')
y_p2 = clf.fit(x_train_res, y_train_res).predict(X_test1)


from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p2)
print('Accuracy: %f' % accuracy)


from lime import lime_text
from sklearn.pipeline import make_pipeline
c2 = make_pipeline(vectorizer, clf)
print(c2.predict_proba([new["comment_text"][0]]))
idx = 0
exp = explainer.explain_instance(X_test[idx], c2.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c2.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])
exp.as_list()
print('Original prediction:', clf.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['article']] = 0
tmp[0,vectorizer.vocabulary_['you']] = 0
print('Prediction removing some features:', clf.predict_proba(tmp)[0,1])
print('Difference:', clf.predict_proba(tmp)[0,1] - clf.predict_proba(X_test1[idx])[0,1])
%matplotlib inline
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('/tmp/oi.html')
exp.show_in_notebook(text=True)
from imblearn.under_sampling import NearMiss
nm = NearMiss()
X_d, y_d = nm.fit_resample(X1, y_train)
print('Resampled dataset shape %s' % Counter(y_d))
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(C=0.1, solver='sag')
scores = cross_val_score(clf1, X_d,y_d, cv=5,scoring='f1_weighted')
y_p3 = clf1.fit(X_d, y_d).predict(X_test1)
from sklearn.metrics import accuracy_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, y_p3)
print('Accuracy: %f' % accuracy)
from lime import lime_text
from sklearn.pipeline import make_pipeline
c3 = make_pipeline(vectorizer, clf1)
print(c3.predict_proba([new["comment_text"][0]]))
idx = 0
exp = explainer.explain_instance(X_test[idx], c3.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c3.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])
exp.as_list()
print('Original prediction:', clf1.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['article']] = 0
tmp[0,vectorizer.vocabulary_['you']] = 0
print('Prediction removing some features:', clf1.predict_proba(tmp)[0,1])
print('Difference:', clf1.predict_proba(tmp)[0,1] - clf1.predict_proba(X_test1[idx])[0,1])
%matplotlib inline
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('/tmp/oi.html')
exp.show_in_notebook(text=True)
#printing ids of comments which are toxic
count=-1
for x in y_test:
    count=count+1
    if x==1:
        print(count)
idx = 141
exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])
exp.as_list()
print('Original prediction:', clf2.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['shit']] = 0
tmp[0,vectorizer.vocabulary_['bastard']] = 0
print('Prediction removing some features:', clf2.predict_proba(tmp)[0,1])
print('Difference:', clf2.predict_proba(tmp)[0,1] - clf2.predict_proba(X_test1[idx])[0,1])
%matplotlib inline
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('/tmp/oi.html')
exp.show_in_notebook(text=True)
idx = 141
exp = explainer.explain_instance(X_test[idx], c2.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c2.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])
exp.as_list()
print('Original prediction:', clf.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['shit']] = 0
tmp[0,vectorizer.vocabulary_['bastard']] = 0
print('Prediction removing some features:', clf.predict_proba(tmp)[0,1])
print('Difference:', clf.predict_proba(tmp)[0,1] - clf.predict_proba(X_test1[idx])[0,1])
%matplotlib inline
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('/tmp/oi.html')
exp.show_in_notebook(text=True)
idx = 141
exp = explainer.explain_instance(X_test[idx], c3.predict_proba, num_features=10)
print('Document id: %d' % idx)
print('Probability(toxic) =', c3.predict_proba([X_test[idx]])[0,1])
print('True class: %s' % class_names[y_test[idx]])
exp.as_list()
print('Original prediction:', clf1.predict_proba(X_test1[idx])[0,1])
tmp = X_test1[idx].copy()
tmp[0,vectorizer.vocabulary_['shit']] = 0
tmp[0,vectorizer.vocabulary_['bastard']] = 0
print('Prediction removing some features:', clf1.predict_proba(tmp)[0,1])
print('Difference:', clf1.predict_proba(tmp)[0,1] - clf1.predict_proba(X_test1[idx])[0,1])
%matplotlib inline
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('/tmp/oi.html')
exp.show_in_notebook(text=True)