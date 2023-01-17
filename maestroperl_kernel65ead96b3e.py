# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import sklearn.model_selection
import math
train_df = pd.read_csv('../input/hse-aml-2020/books_train.csv')
test_df = pd.read_csv('../input/hse-aml-2020/books_test.csv')
sample_df = pd.read_csv('../input/hse-aml-2020/books_sample_submission.csv')


train_df.head(1)
def preprocess(line):
    line = line.strip().lower()
    line = line.translate(str.maketrans('', '', string.punctuation))
    return line
train_df['publisher'] = train_df['publisher'].apply(preprocess)
test_df['publisher'] = test_df['publisher'].apply(preprocess)

vectorizer = HashingVectorizer(n_features=50000)
X = vectorizer.fit_transform(train_df['publisher'])
y = train_df['average_rating']

X_test = vectorizer.transform(test_df['publisher'])

training, valid, ytraining, yvalid = train_test_split(X, y, test_size = 0.5)

model1 = GradientBoostingRegressor()
model1.fit(training, ytraining)
preds1 = model1.predict(valid)
test_preds1 = model1.predict(X_test)
train_df['title'] = train_df['title'].apply(preprocess)
test_df['title'] = test_df['title'].apply(preprocess)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_df['title'])
y = train_df['average_rating']

X_test = vectorizer.transform(test_df['title'])

training, valid, ytraining, yvalid = train_test_split(X, y, test_size = 0.5)

model3 = ElasticNet()
model3.fit(training, ytraining)
preds3 = model3.predict(valid)
test_preds3 = model3.predict(X_test)
train_df['publication_year'] = train_df['publication_date'].map(lambda x: x.split('/')[2])
train_df['publication_month'] = train_df['publication_date'].map(lambda x: x.split('/')[1])
train_df['publication_day'] = train_df['publication_date'].map(lambda x: x.split('/')[0])

test_df['publication_year'] = test_df['publication_date'].map(lambda x: x.split('/')[2])
test_df['publication_month'] = test_df['publication_date'].map(lambda x: x.split('/')[1])
test_df['publication_day'] = test_df['publication_date'].map(lambda x: x.split('/')[0])

train_df['populatity'] = train_df['ratings_count'] / ( train_df['text_reviews_count'] + 0.1 )
test_df['populatity'] = test_df['ratings_count'] / ( test_df['text_reviews_count'] + 0.1 )

X = train_df.drop(['average_rating', 'title','authors', 'isbn', 'isbn13', 'publication_date', 'language_code', 'publisher'], axis = 1)
X_test = test_df.drop(['title','authors', 'isbn', 'isbn13', 'publication_date', 'language_code', 'publisher'], axis = 1)

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
X_test = scaler.transform(X_test)
y = train_df['average_rating']

training, valid, ytraining, yvalid = train_test_split(X, y, test_size = 0.5)


model2 = CatBoostRegressor(iterations=2,
                          learning_rate=1,
                          depth=2)
model2.fit(training, ytraining)
preds2 = model2.predict(valid)
test_preds2 = model2.predict(X_test)
stacking_predictions = np.column_stack((preds1, preds2, preds3))
stacked_test_predictions = np.column_stack((test_preds1, test_preds2, test_preds3))

meta_model = GradientBoostingRegressor()
meta_model.fit(stacking_predictions, yvalid)

final_predictions = meta_model.predict(stacked_test_predictions)
sample_df['average_rating'] = final_predictions
sample_df.to_csv('last1.csv', index=False)


train_df.head()
train_df['populatity'] =  train_df['ratings_count'] / train_df['text_reviews_count']
test_df['populatity'] =  test_df['ratings_count'] / test_df['text_reviews_count']
# 155
for i in range (0, 1000):
    train_df['hundreds'] = train_df['  num_pages'] // i
    print(i, train_df['hundreds'].corr(train_df['  num_pages']), train_df['average_rating'].corr(train_df['hundreds']))
train_df['155pages'] = (train_df['  num_pages'] > 1000).apply(int)
train_df.corr()
train_df['average_rating'].corr(train_df['hundreds'])
n = 700

print(train_df[train_df['  num_pages'] > n].average_rating.mean())
print(train_df[train_df['  num_pages'] > n].average_rating.count())

print(train_df[train_df['  num_pages'] < n].average_rating.mean())
print(train_df[train_df['  num_pages'] < n].average_rating.count())
train_df.gtou

from catboost import CatBoostRegressor
# Initialize data

# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=2,
                          learning_rate=1,
                          depth=2)
# Fit model
model.fit(train_data, train_labels)
# Get predictions
preds = model.predict(eval_data)