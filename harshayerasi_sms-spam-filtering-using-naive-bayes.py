# import pandas

import pandas as pd
# Its not UTF-8 encoded.

df = pd.read_csv('../input/spam.csv',encoding='latin-1')
# Printing first 5 values

df.head()
df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis =1)
df.columns.values
df.head()
df.shape
import numpy as np

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
X = df.v2 # inputs

y = df.v1 # target
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)
print(vectorizer.get_feature_names())
X_train, X_test, y_train, y_test = train_test_split(X,y)
from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile=15)

select.fit(X_train,y_train)
X_train_transformed = select.transform(X_train).toarray()

X_test_transformed = select.transform(X_test).toarray()
X_train_transformed.shape
clf_multi = MultinomialNB()

clf_multi.fit(X_train_transformed,y_train)
predictions = clf_multi.predict(X_test_transformed)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)*100
clf_ber = BernoulliNB()

clf_ber.fit(X_train_transformed,y_train)
predictions = clf_ber.predict(X_test_transformed)

accuracy_score(y_test,predictions)*100
clf_g = GaussianNB()

clf_g.fit(X_train_transformed,y_train)

predictions = clf_g.predict(X_test_transformed)

accuracy_score(y_test,predictions)*100