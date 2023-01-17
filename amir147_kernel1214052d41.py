import numpy as np

import pandas as pd

import scipy.sparse as sp



from sklearn.linear_model import LogisticRegression



from sklearn.metrics import accuracy_score

from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv('train.csv')
data.head()
title_vectorizer = CountVectorizer(max_features=1000, binary=True)

title_features = title_vectorizer.fit_transform(data.title)



descr_vectorizer = CountVectorizer(max_features=1000, binary=True)

description_features = descr_vectorizer.fit_transform(data.description)



features = sp.hstack([title_features, description_features, data[['price']]])
Xtrain, Xval, ytrain, yval = train_test_split(features, data.category_id, 

                                              random_state=241, test_size=0.33)
%%time



clf = LogisticRegression(C=100)

clf.fit(Xtrain, ytrain)

y_pred = clf.predict(Xval)
accuracy_score(yval, y_pred)
test = pd.read_csv('test.csv')
title_features = title_vectorizer.transform(test.title)

description_features = descr_vectorizer.transform(test.description)





features = sp.hstack([title_features, description_features, test[['price']]])
features.shape
y_pred_test = clf.predict(features)
pd.DataFrame({'Id': test.item_id, 'Category':y_pred_test}).to_csv('my_pred.csv', index=None)