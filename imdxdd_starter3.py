import nltk

import os

import re



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import Lasso, Ridge, LinearRegression, SGDRegressor

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy.sparse import hstack

%matplotlib inline
train = pd.read_csv('../input/akvelon2/train.csv')

test = pd.read_csv('../input/akvelon2/test.csv')
sns.distplot(np.log1p(train['score']))
train.head(10)
test.head()
stemmer = nltk.SnowballStemmer('english')

def preprocess(string):

    result = []

    for word in nltk.tokenize.word_tokenize(string):

        if re.search('\w', word):

            result.append(stemmer.stem(word))

    return ' '.join(result)
train['preprocess_review'] = train['review'].apply(preprocess)

test['preprocess_review'] = test['review'].apply(preprocess)
train['preprocess_review'].head()
def cross_validation(data, estimator, vectorizer, n_folds=5):

    # класс кросс валидации

    cv = KFold(n_splits=n_folds, shuffle=True)

    scores = []

    # цикл для обхода фолдов кросс валидации

    for train_idx, valid_idx in cv.split(np.arange(data.shape[0])):

        # выделяем данные для тренировки и валидации

        train_data = data.iloc[train_idx]

        valid_data = data.iloc[valid_idx]

        # кодируем признак name с помощью OneHotEncoder

        encoder = OneHotEncoder(handle_unknown='ignore')

        train_name = encoder.fit_transform(train_data['name'].values.reshape(-1, 1))

        valid_name = encoder.transform(valid_data['name'].values.reshape(-1, 1))

        # кодируем признак preprocess_review с помощью векторизатора из параметра функции

        x_train = vectorizer.fit_transform(train_data['preprocess_review'])

        x_valid = vectorizer.transform(valid_data['preprocess_review'])

        # объеденяем кодированные признаки name и preprocess_review

        x_train = hstack([x_train, train_name])

        x_valid = hstack([x_valid, valid_name])

        # обучаем алгоритм

        estimator.fit(x_train, train_data['score'])

        # делаем предсказание

        predict = estimator.predict(x_valid)

        # добавляем MAE на фолде в массив

        scores.append(mean_absolute_error(valid_data['score'], predict))

    return scores
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=100, max_df=0.4, token_pattern=r'\S+')
cross_validation(train, SGDRegressor(), vectorizer, 5)
estimator = SGDRegressor()

# кодируем признак name с помощью OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')

train_name = encoder.fit_transform(train['name'].values.reshape(-1, 1))

test_name = encoder.transform(test['name'].values.reshape(-1, 1))

# кодируем признак preprocess_review с помощью векторизатора из параметра функции

x_train = vectorizer.fit_transform(train['preprocess_review'])

x_test = vectorizer.transform(test['preprocess_review'])

# объеденяем кодированные признаки name и preprocess_review

x_train = hstack([x_train, train_name])

x_test = hstack([x_test, test_name])

# обучаем алгоритм

estimator.fit(x_train, train['score'])

# делаем предсказание

predict = estimator.predict(x_test)
# создаем признак score в тестовых данных

test['score'] = predict
test[['ID', 'score']].to_csv('sub.csv', index=None)