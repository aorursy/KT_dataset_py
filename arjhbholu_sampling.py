# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/Sampled.csv')
%matplotlib inline
# Any results you write to the current directory are saved as output.
features = df.drop(['Unnamed: 0', 'content', 'recommended', 'author', 'overall_rating'], axis=1)
features2 = df['content']
target = df['recommended']
target.value_counts()
# features.author_country.value_counts()
features['author_country'] = LabelEncoder().fit_transform(features['author_country'])
features['cabin_flown'] = LabelEncoder().fit_transform(features['cabin_flown'])
features['airline_name'] = LabelEncoder().fit_transform(features['airline_name'])
features = StandardScaler().fit_transform(features)
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words='english')
features2 = vectorizer.fit_transform(features2)
features2 = features2.toarray()
merge = np.concatenate((features2, features), axis=1)
merge.shape
# from sklearn.naive_bayes import R
clfs = {'DecisionTree': DecisionTreeClassifier(), 'RandomForestClassifier': RandomForestClassifier(),
       'LogisticRegression': LogisticRegression(), 'SVM': SVC()}
x = []
y = []
z = []
for name, clf in clfs.items():
    x.append({'Classifier': name, 'Features': 'Features 1', 'value': cross_val_score(clf, features, target, cv=5).mean()})
    y.append({'Classifier': name, 'Features': 'Features 2', 'value': cross_val_score(clf, features2, target, cv=5).mean()})
    z.append({'Classifier': name, 'Features': 'Features 3', 'value': cross_val_score(clf, merge, target, cv=5).mean()})
    print (name)
#     print ('{0} \t {1} \t {2}'.format(x, y, z))
result = pd.DataFrame(columns=['Classifier', 'Features', 'value'])
for i in range(4):
    result = result.append(x[i], ignore_index=True)
    result = result.append(y[i], ignore_index=True)
    result = result.append(z[i], ignore_index=True)
result
ax, fig = plt.subplots(figsize=(10, 10))
sns.pointplot(x="Classifier", y="value", hue="Features", data=result)
clfs = {'DecisionTree': DecisionTreeClassifier(), 'RandomForestClassifier': RandomForestClassifier(),
       'LogisticRegression': LogisticRegression(), 'SVM': SVC()}
for name, clf in clfs.items():
    originalClass = []
    predictedClass = []
    score = []
    cv = StratifiedKFold(n_splits=5)
    print (name)
    for tr, ts in cv.split(features2, target):
        X_train = features2[tr]
        y_train = target[tr]
        X_test = features2[ts]
        y_test = target[ts]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        originalClass.extend(y_test)
        predictedClass.extend(y_pred)
        score.append(accuracy_score(y_test, y_pred))
    print (sum(score)/float(len(score)))
    print (classification_report(originalClass, predictedClass))
clfs = {'DecisionTree': DecisionTreeClassifier(), 'RandomForestClassifier': RandomForestClassifier(),
       'LogisticRegression': LogisticRegression(), 'SVM': SVC()}
for name, clf in clfs.items():
    originalClass = []
    predictedClass = []
    score = []
    cv = StratifiedKFold(n_splits=5)
    print (name)
    for tr, ts in cv.split(features, target):
        X_train = features[tr]
        y_train = target[tr]
        X_test = features[ts]
        y_test = target[ts]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        originalClass.extend(y_test)
        predictedClass.extend(y_pred)
        score.append(accuracy_score(y_test, y_pred))
    print (sum(score)/float(len(score)))
    print (classification_report(originalClass, predictedClass))
clfs = {'DecisionTree': DecisionTreeClassifier(), 'RandomForestClassifier': RandomForestClassifier(),
       'LogisticRegression': LogisticRegression(), 'SVM': SVC()}
for name, clf in clfs.items():
    originalClass = []
    predictedClass = []
    score = []
    cv = StratifiedKFold(n_splits=5)
    print (name)
    for tr, ts in cv.split(merge, target):
        X_train = merge[tr]
        y_train = target[tr]
        X_test = merge[ts]
        y_test = target[ts]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        originalClass.extend(y_test)
        predictedClass.extend(y_pred)
        score.append(accuracy_score(y_test, y_pred))
    print (sum(score)/float(len(score)))
    print (classification_report(originalClass, predictedClass))
mer.shape