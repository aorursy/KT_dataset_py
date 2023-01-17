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
 # отключим предупреждения

import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import classification_report



from nltk import word_tokenize  

from nltk.stem.snowball import PorterStemmer
# Загружаем данные

train = pd.read_csv('/kaggle/input/simplesentiment/products_sentiment_train.tsv', sep = '\t', header = None, names = ['text', 'y'])

test = pd.read_csv('/kaggle/input/simplesentiment/products_sentiment_test.tsv', sep = '\t')
print ("Количество размеченных отзывов: %d" % (train.shape[0]))

print ("Количество позитивных отзывов: %d (%0.1f%%)" % (train.y.sum(), 100.*train.y.mean()))

print ("Количество тестовых отзывов: %d" % (test.shape[0]))
# Пример нескольких отзывов

pd.set_option('max_colwidth', 300)

train.head()
# Создадим вспомогательных аналайзер на основе стеммера Портера

stemmer = PorterStemmer()

analyzer = TfidfVectorizer().build_analyzer()



def stemmed(text):

    return (stemmer.stem(word) for word in analyzer(preprocess(text)))
# Заменим 't на not

def preprocess(text):

    return text.replace(" 't", " not")



train['x'] = train.text.apply(preprocess)

test['x'] = test.text.apply(preprocess)
# Объединим векторизованные фичи разных типов токенов

union = FeatureUnion([("word11", TfidfVectorizer(ngram_range=(1,1), analyzer='word')),

                      ("stem11", TfidfVectorizer(ngram_range=(1,1), analyzer=stemmed)),

                      ("word23", TfidfVectorizer(ngram_range=(2,3), analyzer='word')),

                      ("stem23", TfidfVectorizer(ngram_range=(2,3), analyzer=stemmed)),

                      ("char14", TfidfVectorizer(ngram_range=(1,4), analyzer='char'))])



# Объединим в Pipeline с линейной регрессией в качестве классификатора

pipe = Pipeline([("vectorizer", union),

                 ("classifier", LogisticRegression(penalty = 'l2'))])



# Расчитаем точность по кроссвалидации

scores = cross_val_score(pipe, train.x, train.y, cv = 5)



print ("Средняя точность: %0.2f%%" % (100.*scores.mean()))

print ("Среднеквадратичное отклонение: %0.4f" % scores.std())
# Посмотрим на ошибки

X_train, X_test, y_train, y_test = train_test_split(train.x, train.y, test_size=0.2, random_state=0)

pipe.fit(X_train, y_train)



y_pred = pipe.predict(X_test)

p_test = pipe.predict_proba(X_test)

check = pd.DataFrame(X_test)

check['y'] = y_test

check['y_pred'] = y_pred

check['p0'] = p_test[:,0]

check['p1'] = p_test[:,1]



#check.head()

print(classification_report(y_test, y_pred))
# Обучим классификатор на всех размененных данных

pipe.fit(train.x, train.y)

test['y'] = pipe.predict(test.x)



# Запишем в файл решение для загрузки на Kaggle

test[['Id','y']].to_csv('product-reviews-sentiment-analysis-light.csv', index = False)

# Проверим, что записалось корректно

! head -5 product-reviews-sentiment-analysis-light.csv