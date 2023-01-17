# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

from stop_words import get_stop_words



import warnings

warnings.filterwarnings('ignore')
def print_scores(y_valid, y_pred):

    '''

    Функция для быстрого вывода четырёх метрик для регрессии.

    

    y_valid --- истинные значения

    y_pred --- предсказанные моделью значения

    '''

    print('MSE:', mean_squared_error(y_valid, y_pred))

    print('MAE:', mean_absolute_error(y_valid, y_pred))

    print('MedAE:', median_absolute_error(y_valid, y_pred))

    print('R2:', r2_score(y_valid, y_pred))
def model_fit_and_predict(X, y, model, **params):

    '''

    Разбиение на train и valid, 

    обучение и предсказание с помощью модели, 

    а также вывод результатов.

    '''

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)

    

    my_model = model(**params)

    my_model.fit(X_train, y_train)

    y_pred = my_model.predict(X_valid)

    

    print_scores(y_valid, y_pred)

    

    return my_model
df = pd.read_csv('../input/dou-jobs/df_26-12-2019.csv', index_col=0)

df_1 = df[['name', 'description', 'salary']]

df_1['log_salary'] = np.log(df_1['salary'])

df_1.head()
stop_words_ukr = get_stop_words('ukrainian')

stop_words_rus = get_stop_words('russian')

stop_words_ukr_rus = list(set(stop_words_ukr) | set(stop_words_rus))
bow = CountVectorizer(stop_words=stop_words_ukr_rus, min_df=5)

tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, min_df=5)



X_name_bow = bow.fit_transform(df_1['name'])

X_name_tfidf = tfidf.fit_transform(df_1['name'])



y = df_1['log_salary']
model_fit_and_predict(X_name_bow, y, LinearRegression)
my_model = model_fit_and_predict(X_name_bow, y, ElasticNet, l1_ratio=0)
enet_params = {'l1_ratio': np.linspace(0.1, 0.9, 9)}



enet_grid = GridSearchCV(ElasticNet(), enet_params, cv=5, scoring='neg_mean_squared_error')

enet_grid.fit(X_name_bow, y)
enet_grid.best_params_
enet_grid.best_score_
enet_params = {'alpha': np.logspace(-4, -3, 5)}



enet_grid = GridSearchCV(ElasticNet(), enet_params, cv=5, scoring='neg_mean_squared_error')

enet_grid.fit(X_name_bow, y)
enet_grid.best_score_
enet_grid.best_params_
enet_params = {'l1_ratio': np.linspace(0.1, 0.9, 9)}



enet_grid = GridSearchCV(ElasticNet(alpha=0.00017782794100389227), 

                         enet_params, cv=5, scoring='neg_mean_squared_error')

enet_grid.fit(X_name_bow, y)
enet_grid.best_score_
my_model_1 = model_fit_and_predict(X_name_tfidf, y, LinearRegression)
my_model_1 = model_fit_and_predict(X_name_tfidf, y, ElasticNet, l1_ratio=0.0)
enet_params = {'l1_ratio': np.linspace(0.1, 0.9, 9)}



enet_grid_1 = GridSearchCV(ElasticNet(), enet_params, cv=5, scoring='neg_mean_squared_error')

enet_grid_1.fit(X_name_tfidf, y)
# bow = CountVectorizer(stop_words=stop_words_ukr_rus, ngram_range=(1, 2), min_df=5)

# tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus, ngram_range=(1, 2), min_df=5)



# X_name_bow = bow.fit_transform(df_1['name'])

# X_name_tfidf = tfidf.fit_transform(df_1['name'])



# y = df_1['log_salary']
# model_fit_and_predict(X_name_bow, y, LinearRegression)
# model_fit_and_predict(X_name_tfidf, y, LinearRegression)