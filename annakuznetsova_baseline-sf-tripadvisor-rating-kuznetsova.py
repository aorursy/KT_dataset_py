import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
df_train.head(5)
df_test.info()
df_test.head(5)
sample_submission.head(5)
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
df.info()
df.Reviews[1]
# приведем Restaurant_id только к числовому значению

df['Restaurant_id'] = df['Restaurant_id'].apply(lambda x: (x[3:]))
# Посмотрим, какие значения хранятся в данном признаке

df['Cuisine Style']
# заполним пропуски нулями, а заполненные ячейки единицами 

df['Cuisine Style'].fillna(0, inplace = True)

df['Cuisine Style'] = df['Cuisine Style'].apply(lambda x: x if x == 0 else 1)
# посмотрим, какие значения хранятся в данном признаке 

df['Price Range'].value_counts(dropna=False)
# видим большое количество пропусков, но у большей части ресторанов средний уровень цен. 

# добавим признак с информацией о пропусках, а затем заполним пропуски в столбце 'Price Range' на среднее зачение



df['NaN_Price Range'] = pd.isna(df['Price Range']).astype('float64') 



df['Price Range'].fillna('$$ - $$$', inplace = True)
df['Price Range'].value_counts()
# применим к столбцу 'Price Range' функцию get_dummies

df = pd.get_dummies(df, columns=[ 'Price Range',], dummy_na=True)
# сохраним информацию о пропущенных значениях в новом признаке

df['Number_of_Reviews_isNAN'] = pd.isna(df['Number of Reviews']).astype('uint8')
df_train['Number of Reviews'].describe()
# построим boxplot для признака 'Number of Reviews'

sns.boxplot(df['Number of Reviews'])
# заполняем пропуски 0

df['Number of Reviews'].fillna(0, inplace=True)
# в признаке 'Reviews' пропуска. Заменим их на пустую строку, которая встречается в признаке

df['Reviews'] = df['Reviews'].fillna('[[], []]')
# сохраним информацию о пустых строках в отдельный признак 

df['Reviews_Nan'] = (df['Reviews']=='[[], []]').astype('float64')
# созданим признак 'date_of_Review', где будут храниться только даты отзывов

df['date_of_Review'] = df['Reviews'].str.findall('\d+/\d+/\d+')
# разделим даты на два отдельных признака 

df['date_1'] = df.date_of_Review.map(lambda x: str(x)[2:12])

df['date_2'] = df.date_of_Review.map(lambda x: str(x)[-12:-2])

df['date_1']= pd.to_datetime(df['date_1'])

df['date_2']= pd.to_datetime(df['date_2'])



# и создадим признак с разницей этих дат 

df['delta_date'] = df['date_1'] - df['date_2']
df['delta_date'].fillna(0, inplace=True)
# переведем признак 'delta_date' в int64

df['delta_date'] = df['delta_date'].dt.days.astype('int64')
# и умножим значения меньше нуля на (-1)

df['delta_date'] = df['delta_date'].apply(lambda x: x if x > 0 else x*(-1))
# применим к столбцу 'City' функцию get_dummies

df = pd.get_dummies(df, columns=[ 'City',], dummy_na=True)
df.columns
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
# У нас много ресторанов, которые не дотягивают и до 2500 места в своем городе, а что там по городам?

df_train['City'].value_counts(ascending=True).plot(kind='barh')
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
df['ID_TA'] = df['ID_TA'].apply(lambda x: float(x[1:]))
df['URL_TA'] = df['URL_TA'].str.split('-').apply(lambda x: x[1][1:]).astype('float64')
# Теперь выделим тестовую часть

train_data = df.query('sample == 1').drop(['sample'], axis=1)

test_data = df.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating', 'Reviews', 'date_of_Review', 'date_1', 'date_2'], axis=1)
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# проверяем

test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating', 'Reviews', 'date_of_Review', 'date_1', 'date_2'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)