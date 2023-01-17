import numpy as np
import pandas as pd
import os
import re
import xgboost
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import unidecode
import nltk
from sklearn.neighbors import KNeighborsClassifier
print(os.listdir("../input"))
print(os.listdir())
lemmatizer = nltk.WordNetLemmatizer()
#read data
train_df = pd.read_json("../input/train.json")
test_df = pd.read_json("../input/test.json")
Y_train = train_df["cuisine"].apply(lambda x: x.lower())
X_train = train_df["ingredients"].apply(lambda x: ' '.join(lemmatizer.lemmatize(unidecode.unidecode(i)) for i in x).strip().lower())
test_data = test_df["ingredients"].apply(lambda x: ' '.join(lemmatizer.lemmatize(unidecode.unidecode(i)) for i in x).strip().lower())
k = Y_train.nunique()
#label encoding
encoder = LabelEncoder()
y_target = encoder.fit_transform(Y_train)
#vectorizing
vectorizer = TfidfVectorizer(binary = True)
x = vectorizer.fit_transform(X_train.values)
x_test = vectorizer.transform(test_data.values)
#SVC implementation
model = SVC(C=270, kernel='rbf', degree=3, gamma=1.3, coef0=1.0, shrinking=True, tol=0.001, probability=True, cache_size=500, max_iter=-1)
ovr = OneVsRestClassifier(model, n_jobs=1)
ovr.fit(x, y_target)
ovr_pred = ovr.predict(x_test)
y_test = encoder.inverse_transform(ovr_pred)
test_id = test_df["id"]
submit_cook = pd.DataFrame({'id': test_id, 'cuisine': y_test}, columns=['id', 'cuisine'])
submit_cook.to_csv('svc_tf.csv', index=False)
#XGBoost implementation
model = xgboost.XGBClassifier(max_depth = 9, eta = 0.003, subsample = 0.7, gamma = 7)
ovr = OneVsRestClassifier(model, n_jobs = -1)
ovr.fit(x, y_target)
ovr_pred = ovr.predict(x_test)
y_test = encoder.inverse_transform(ovr_pred)
test_id = test_df["id"]
submit_xg = pd.DataFrame({'id': test_id, 'cuisine': y_test}, columns=['id', 'cuisine'])
submit_xg.to_csv('xgboost_tf.csv', index=False)