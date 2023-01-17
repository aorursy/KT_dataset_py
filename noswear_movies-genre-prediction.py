import pandas as pd
ls
data = pd.read_csv("../input/train.csv")
data.head()
%time data['genre'].value_counts()
data['label'] = data.genre.map({'drama':0,'thriller':1,'comedy':2,'action':3,'sci-fi':4,'horror':5,'other':6,'adventure':7,'romance':8})
data.tail()
X = data.text

y = data.label
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =  train_test_split(X,y,random_state=1)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english',ngram_range=(1,2),max_df=0.5)
vect.fit(X_train)

X_train_dtm = vect.transform(X_train)
X_train_dtm
X_test_dtm = vect.transform(X_test)

X_test_dtm
# Multinomial Naive Bayes



from sklearn.naive_bayes import MultinomialNB

nb= MultinomialNB()
%time nb.fit(X_train_dtm,y_train)
y_pred_class = nb.predict(X_test_dtm)
from sklearn import metrics

metrics.accuracy_score(y_test,y_pred_class)
# Logistics Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
%time logreg.fit(X_train_dtm,y_train)
y_pred_log = logreg.predict(X_test_dtm)

metrics.accuracy_score(y_test,y_pred_log)
# Stochastic gradient descent

from sklearn.linear_model import SGDClassifier

sgdc = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
%time sgdc.fit(X_train_dtm,y_train)
y_pred_sgd = sgdc.predict(X_test_dtm)

metrics.accuracy_score(y_test,y_pred_sgd)
# Random Forest



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
%time rfc.fit(X_train_dtm,y_train)
y_pred_rfc = rfc.predict(X_test_dtm)
y_pred_rfc = rfc.predict(X_test_dtm)

metrics.accuracy_score(y_test,y_pred_rfc)