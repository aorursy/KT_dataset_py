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

from scipy.sparse import hstack



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
df = pd.read_csv('../input/dou-jobs/df_26-12-2019.csv', index_col=0)

df_1 = df[['name', 'description', 'salary']]

df_1['log_salary'] = np.log(df_1['salary'])

df_1.head()
stop_words_ukr = get_stop_words('ukrainian')

stop_words_rus = get_stop_words('russian')

stop_words_ukr_rus = list(set(stop_words_ukr) | set(stop_words_rus))
tfidf = TfidfVectorizer(stop_words=stop_words_ukr_rus,  ngram_range=(1, 2))



name_tfidf = tfidf.fit_transform(df_1['name'])

desc_tfidf = tfidf.fit_transform(df_1['description'])



X = hstack([name_tfidf, desc_tfidf])



y = df_1['log_salary']
model_fit_and_predict(X, y, LinearRegression)
# model_fit_and_predict(X, y, ElasticNet, l1_ratio=0)