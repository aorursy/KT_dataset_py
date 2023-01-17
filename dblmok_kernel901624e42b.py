# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import xgboost as xgb
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
with open('../input/train.json', 'r', encoding='utf-8') as fh: #открываем файл на чтение
    data_train = json.load(fh) #загружаем из файла данные в словарь data
with open('../input//test.json', 'r', encoding='utf-8') as fh1: #открываем файл на чтение
    data_test = json.load(fh1) #загружаем из файла данные в словарь data
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# функция суммы строковых массивов
def arr_str_sum(arr_str):
    temp = ''
    temp_ = ' '
    for i in range(0, len(arr_str)):
        temp = temp + arr_str[i] + temp_
    return temp
# функция суммы строковых массивов

#совпадение целевого вектора
def coincidence(y, y_pred):
    count = 0
    for i in range(0,len(y_pred)):
        t = 1 if y[i]==y_pred[i] else 0
        count = count+t
    return count/len(y_pred)
#совпадение целевого вектора

#лемматизация составляющих рецепт
lemmatizer = WordNetLemmatizer()
def preprocess(ingredients):
    ingredients_text = ' '.join(ingredients)
    ingredients_text = ingredients_text.lower()
    ingredients_text = ingredients_text.replace('-', ' ')
    words = []
    for word in ingredients_text.split():
        if re.findall('[0-9]', word): continue
        if len(word) <= 2: continue
        if '’' in word: continue
        word = lemmatizer.lemmatize(word)
        if len(word) > 0: words.append(word)
    return ' '.join(words)
#лемматизация составляющих рецепт

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# НАЧИНАЕМ ГЕНЕРАЦИЮ ПРИЗНАКОВ№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№
###############################################################################################


# добавляем признак количество ингридиентов
for i in range(0,len(data_train)):
    data_train[i].setdefault("Amount of ingredients", len(data_train[i]['ingredients']))

for i in range(0,len(data_test)):
    data_test[i].setdefault("Amount of ingredients", len(data_test[i]['ingredients']))

# добавляем признак количество ингридиентов

#одна большая строка-рецепт
for i in range(0,len(data_train)):
    data_train[i].setdefault("All ingredients in one string", preprocess(data_train[i]['ingredients']))
    
for i in range(0,len(data_test)):
    data_test[i].setdefault("All ingredients in one string", preprocess(data_test[i]['ingredients']))
#одна большая строка-рецепт


# добавляем индикаторы того, что в названии ингридиентов есть название страны
# некоторые страны убрали, так как их названия практически не встречаются
country = ['italian','mexican','chinese','french',
           'thai','japanese','greek','spanish','korean','vietnamese' ,'jamaican']

from itertools import product

for i, c in product(range(0,len(data_train)), country):
        ind = 0 if data_train[i]['All ingredients in one string'].find(c) < 0 else 1
        data_train[i].setdefault(c, ind)
for i, c in product(range(0,len(data_test)), country):
        ind = 0 if data_test[i]['All ingredients in one string'].find(c) < 0 else 1
        data_test[i].setdefault(c, ind)
# добавляем индикаторы того, что в названии ингридиентов есть название страны
all_recipe = []
for i in range(0,len(data_train)):
    all_recipe.append(data_train[i]['All ingredients in one string'])
for i in range(0,len(data_test)):
    all_recipe.append(data_test[i]['All ingredients in one string'])
    
# генерим матрицу TFIDF
vectorizer = CountVectorizer()
Mat_count = vectorizer.fit_transform(all_recipe)
matrix_freq = np.asarray(Mat_count.sum(axis=0)).ravel()
transformer = TfidfTransformer()
Mat_count = transformer.fit_transform(Mat_count)
Names_ingridients = vectorizer.get_feature_names()

df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)
del(df_train['All ingredients in one string'])
del(df_train['ingredients'])
del(df_test['All ingredients in one string'])
del(df_test['ingredients'])
pdMat = pd.DataFrame(Mat_count.toarray())
y_train= df_train.cuisine.replace(
        {'greek': 1, 'southern_us': 2, 'filipino': 3, 'indian': 4, 'jamaican': 5, 'spanish': 6, 'italian': 7,
 'mexican': 8, 'chinese': 9, 'british': 10, 'thai': 11, 'vietnamese': 12, 'cajun_creole': 13,
 'brazilian': 14, 'french': 15, 'japanese': 16, 'irish': 17, 'korean': 18, 'moroccan': 19, 'russian': 20})

df_train = pd.concat([df_train, pdMat[:39774]], axis=1)
df_id = df_train['id']
del(df_train['id'])
del(df_train['cuisine'])
df_mat = pdMat[39774:]
df_mat.index = range(0,9944)
df_test = pd.concat([df_test, df_mat], axis=1)
df_id_test =  df_test['id']
del(df_test['id'])
# Any results you write to the current directory are saved as output.
scaler = StandardScaler()
scaler.fit(df_train)
df_train = scaler.transform(df_train)
scaler.fit(df_test)
df_test = scaler.transform(df_test)
from sklearn.svm import SVC
estimator = SVC(
    C=50,
    kernel='rbf',
    gamma=1.4,
    coef0=1,
    cache_size=500,
)
cl = OneVsRestClassifier(estimator, n_jobs=-1)
cl.fit(df_train, y_train)
y_pred = cl.predict(df_train)
print(coincidence(y_train, y_pred))
y_pred = cl.predict(df_test)
y_pred= pd.DataFrame(y_pred).replace(
        {1:'greek', 2:'southern_us', 3:'filipino', 4:'indian', 5:'jamaican', 6:'spanish', 7:'italian',
 8:'mexican', 9:'chinese', 10:'british', 11:'thai', 12:'vietnamese', 13:'cajun_creole',
 14:'brazilian', 15:'french', 16:'japanese', 17:'irish', 18:'korean', 19:'moroccan', 20:'russian'})
y_pred.index = df_id_test
y_pred.columns  = ["cuisine"]
y_pred.to_csv('predict1.csv', header=True, index_label='Id')