import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
wine_base = pd.read_csv(os.path.join('../input/', 'winemag-data-130k-v2.csv'))
wine_base.head(5)
from sklearn.model_selection import train_test_split
X = wine_base.drop(['points'], axis=1)
y = wine_base['points'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)
vect = CountVectorizer(min_df=5)
vect.fit(X_train['description'])
print("vocabulary size: {}".format(len(vect.vocabulary_)))
X_train_vectored = vect.transform(X_train['description'])
feature_names = vect.get_feature_names()
print("Number of Features: ", len(feature_names))
print("First 20 features: \n{}".format(feature_names[:20]))
print("features 10010 to 10030:\n{}".format(feature_names[10010:10030]))
print("Every 2000th feature:\n{}".format(feature_names[::2000]))
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV

lr = LinearRegression()
lr.fit(X_train_vectored, y_train)

from sklearn.metrics import mean_squared_error
y_pred = lr.predict(X_train_vectored)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
rmse
X_test_vectored = vect.transform(X_test['description'])
y_test_pred = lr.predict(X_test_vectored)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
rmse