# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import nltk
import string
import re
import numpy as np
import pandas as pd
import pickle
#import lda

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_notebook
#from bokeh.transform import factor_cmap

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("lda").setLevel(logging.WARNING)
PATH = "../input/"
train = pd.read_csv(f'{PATH}train.tsv', sep='\t')
test = pd.read_csv(f'{PATH}test.tsv', sep='\t')
# different data types in the dataset: categorical (strings) and numeric
print(train.dtypes)
print(train.head())

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.brand_name.astype('U'))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, train['price'].astype(int))
docs_new = test.brand_name.astype('U')
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted1 = clf.predict(X_new_tfidf)
print("predicted 1")
print(predicted1)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.name.astype('U'))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, train['price'].astype(int))
docs_new = test.name.astype('U')
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted2 = clf.predict(X_new_tfidf)
print("predicted 2")
print(predicted2)

Xtrain=train.item_condition_id.astype(int)
Ytrain=train.price.astype(int)
Xtest=test.item_condition_id.astype(int)

from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()
regr.fit(Xtrain[:, np.newaxis], Ytrain)
predicted5 = regr.predict(Xtest[:, np.newaxis])
print("predicted 5")
print(predicted5)


from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)), ])
text_clf_svm.fit(train.item_description.astype('U'), train['price'].astype(int))
predicted7 = text_clf_svm.predict(test.item_description.astype('U'))

train_category_code = train.category_name.astype('category')
test_category_code = test.category_name.astype('category')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

m = RandomForestRegressor(n_jobs=-1,min_samples_leaf=3,n_estimators=200)
m.fit(train_category_code.cat.codes[:, np.newaxis], Ytrain)
predicted8 = m.predict(test_category_code.cat.codes[:, np.newaxis])

predicted = (predicted1+predicted2+predicted5+predicted7+predicted8)/5



submission = test[["test_id"]]
submission["price"] = predicted
print(submission)
submission.to_csv("./sample_submission.csv", index=False)