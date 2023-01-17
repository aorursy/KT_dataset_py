# импортируем необходимые для начала библиотеки

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# загрузим данные из .csv файла в таблицу

data = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')

df = data.copy()

df.head()
# посмотрм на расперделение числовых данных в таблице

df.describe()
# посмотрим основную информацию о таблице

df.info()
# визуально посмотрим на распределение незаполненных значений по признакам

fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(df.isnull(), cbar=False, cmap="Greys_r")

plt.show()
# посмотрим на количество строк в таблице

len(df)
# удалим признаки с наибольшим количеством незаполненных данных, 

# а также не влияющие на итоговую классификацию

df.drop(columns=['id', 'url', 'region', 'region_url', 

                 'vin', 'size', 'image_url', 

                 'description', 'county', 

                 'state', 'lat', 'long'], inplace=True)
from sklearn.impute import SimpleImputer



# заменим пропуски в столбце 'odometer' средним значением по столбцу

imr = SimpleImputer(strategy='mean')

imr = imr.fit(df[['odometer']])

imputed_data = imr.transform(df[['odometer']])

df['odometer'] = pd.DataFrame(imputed_data)

df['odometer'].head()
# удалим строки с пропусками в данных

df.dropna(inplace=True)
# ещё раз визуально проверим распределение незаполненных значений по признакам

fig, ax = plt.subplots(figsize=(14,10))

sns.heatmap(df.isnull(), cbar=False, cmap="Greys_r")

plt.show()
# посмотрим на значения в целевой переменной

df['price'].value_counts
# удалим из таблицы объекты с экстремальными значениями по целевой переменной

df = df[df['price'] > 1000]

df = df[df['price'] < 50000]
# посмотрим на количество оставшихся строк после обработки

len(df)
# посмотрим на получившуюся в итоге таблицу с данными

df.head()
from sklearn import preprocessing



# создадим функцию, кодирующую числовыми значениями категориальные признаки

def encode_features(dataframe):

    result = dataframe.copy()

    encoders = {}

    for column in result.columns:

        if result.dtypes[column] == np.object:

            encoders[column] = preprocessing.LabelEncoder()

            result[column] = encoders[column].fit_transform(result[column])

    return result, encoders



# обработаем все столбцы кроме 'description'

encoded_df, encoders = encode_features(df) 

encoded_df.head()
# построим гистограммы различных признаков для оценки корректности данных

encoded_df.hist(figsize=(18, 8), layout=(3,5), bins=20)

print('Features\' hists plotted')
# построим на матрице корреляций зависимость между признаками, а также между признаками и целевой переменной

plt.subplots(figsize=(17, 15))

sns.heatmap(encoded_df.corr(), square = True, annot=True)

plt.show()
# отделим целевую переменную от признаков

y = np.array(encoded_df['price'])

del encoded_df['price']

X = encoded_df.values

X.shape, y.shape
from sklearn.model_selection import train_test_split



# разобьём данные на обучающие и испытательные наборы

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler



# стандартизируем значения признаков

stdsc = StandardScaler()

X_train = stdsc.fit_transform(X_train)

X_test = stdsc.transform(X_test)
%%time

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(X_train, y_train)
# делаем предсказания и выводим метрики

preds = linreg.predict(X_test)

print('R2 linreg: ', r2_score(y_test, preds))

print('MAE linreg: ', mean_absolute_error(y_test, preds))
%%time

from sklearn.linear_model import SGDRegressor



sgd = SGDRegressor()

sgd.fit(X_train, y_train)
# делаем предсказания и выводим метрики

preds = sgd.predict(X_test)

print('R2 sgd: ', r2_score(y_test, preds))

print('MAE sgd: ', mean_absolute_error(y_test, preds))
%%time

from sklearn.tree import DecisionTreeRegressor



tree = DecisionTreeRegressor()

tree.fit(X_train, y_train)
# делаем предсказания и выводим метрики

preds = tree.predict(X_test)

print('R2 tree: ', r2_score(y_test, preds))

print('MAE tree: ', mean_absolute_error(y_test, preds))
%%time

from sklearn.ensemble import GradientBoostingRegressor



gbreg = GradientBoostingRegressor()

gbreg.fit(X_train, y_train)
# делаем предсказания и выводим метрики

preds = gbreg.predict(X_test)

print('R2 gbreg: ', r2_score(y_test, preds))

print('MAE gbreg: ', mean_absolute_error(y_test, preds))
# зарузим tensorboard для визуализации поцессов обучения

!jupyter nbextension enable --py widgetsnbextension

%load_ext tensorboard
%tensorboard --logdir logs



from sklearn.model_selection import GridSearchCV

import xgboost as xgb



# переберём параметры XGBRegressor с помощью GridSearchCV

# визуализируем процесс обучения с помощью tensorboard

alg = xgb.XGBRegressor() 

grid = {'n_estimators': [60, 100, 120, 140], 

        'learning_rate': [0.01, 0.1],

        'max_depth': [5, 7],

        'reg_lambda': [0.5]}

gs = GridSearchCV(estimator=alg, param_grid=grid, cv=5, n_jobs=-1)

%time gs.fit(X_train, y_train)

print('Best score: ', gs.best_score_)

print('Best parameters: ', gs.best_params_)
# инициализируем алгоритм с лучшими параметрами и обучаем модель

xgbreg = gs.best_estimator_

xgbreg.fit(X_train, y_train)
# делаем предсказания и выводим метрики

preds = xgbreg.predict(X_test)

print('R2 xgbreg: ', r2_score(y_test, preds))

print('MAE xgbreg: ', mean_absolute_error(y_test, preds))
%%time

from sklearn.ensemble import RandomForestRegressor 



rnfst = RandomForestRegressor()

rnfst.fit(X_train, y_train)
# делаем предсказания и выводим метрики

preds = rnfst.predict(X_test)

print('R2 rnfst: ', r2_score(y_test, preds))

print('MAE rnfst: ', mean_absolute_error(y_test, preds))
# посмотрим на гистограмме на распределение абсолютной ошибки

plt.hist(y_test - preds)
# наложим друг на друга гистограммы результатов предсказаний и истинных значений

plt.hist(y_test)

plt.hist(preds)
import pickle



# сохраняем модель случайных лесов

file = open('RandomForest_model.pickle','wb')

pickle.dump(rnfst, file)

file.close()