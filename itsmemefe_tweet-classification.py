import pandas as pd

from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

import numpy as np

from sklearn.svm import SVC
df = pd.read_csv("/kaggle/input/tweets-of-trump-and-trudeau/tweets.csv")

df.head()
X = df["status"]

y = df["author"]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=52, test_size=0.33)
cVec = CountVectorizer(stop_words='english', min_df=.05, max_df=.9)

tfidfVec = TfidfVectorizer(stop_words='english', min_df=.05, max_df=.9)



cv_train = cVec.fit_transform(X_train)

cv_test = cVec.transform(X_test)



tv_train = tfidfVec.fit_transform(X_train)

tv_test = tfidfVec.transform(X_test)
params = {

    'min_samples_leaf': np.arange(1,10,1),

    'criterion': ["gini", "entropy"],

    'max_depth': np.arange(5,50,5)

}
gcv = GridSearchCV(DecisionTreeClassifier(random_state=0), params)

gcv.fit(cv_train, y_train)

print(gcv.best_params_)
dt = DecisionTreeClassifier(random_state=0, max_depth=10, criterion= 'gini')

dt.fit(cv_train, y_train)

cv_pred = dt.predict(cv_test)



dt.fit(tv_train, y_train)

tv_pred = dt.predict(tv_test)
print(metrics.accuracy_score(cv_pred,y_test))

print(metrics.confusion_matrix(y_test,cv_pred))

print(metrics.classification_report(y_test,cv_pred))



print(metrics.accuracy_score(tv_pred,y_test))

print(metrics.confusion_matrix(y_test,tv_pred))

print(metrics.classification_report(y_test,tv_pred))
svc_model = SVC(kernel='poly', degree=3)



svc_model.fit(cv_train, y_train)

cv_pred = svc_model.predict(cv_test)



svc_model.fit(tv_train, y_train)

tv_pred = svc_model.predict(tv_test)
print(metrics.accuracy_score(cv_pred,y_test))

print(metrics.confusion_matrix(y_test,cv_pred))

print(metrics.classification_report(y_test,cv_pred))



print(metrics.accuracy_score(tv_pred,y_test))

print(metrics.confusion_matrix(y_test,tv_pred))

print(metrics.classification_report(y_test,tv_pred))