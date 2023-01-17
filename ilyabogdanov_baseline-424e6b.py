# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

pred=pd.read_csv('../input/random_prediction.csv')
y_train = pd.read_csv('../input/y_train.csv')
y_train = y_train['Probability']

with open('../input/x_train.txt', 'r') as f:
    x_train = f.readlines()
with open('../input/x_test.txt', 'r') as f:
    x_test = f.readlines()
x_train = pd.DataFrame(x_train, columns=['reviews'])['reviews']
x_test = pd.DataFrame(x_test, columns=['reviews'])['reviews']


# x_tr, x_tst, y_tr, y_tst = train_test_split(x_train, y_train, train_size=0.8, random_state=42)
print("Finished")
vectorizer = TfidfVectorizer(encoding='utf8', 
                             min_df=5, 
                             ngram_range=(1, 2), 
                             max_features=10000)
vectorizer.fit(x_train[0:int(1e6)])
print('Finished')
X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)
print(X_train.shape, X_test.shape)
print('Finished')

lr = LogisticRegression()
lr.fit(X_train, y_train)
preds = lr.predict_proba(X_test)[:,1]
print('Finished')
print(preds)
pred=pd.read_csv('../input/random_prediction.csv')
pred['Probability']=preds
print(pred)
pred.to_csv('pred.csv',index=False)