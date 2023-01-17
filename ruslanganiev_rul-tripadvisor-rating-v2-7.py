import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

pattern=re.compile('\d+')

import datetime

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
sred = data.groupby(['City'])['Number of Reviews'].median()

data['Number of Reviews'].fillna(0, inplace=True)

spis = list(df_train.City.unique())

nasel = [2140,961,8787,3601,1456,237,1366,429,1840,2872,1620,3223,553,1198,428,1758,1749,615,859,515,1830,505,

       1280,673,643,488,200,284,655,602,766]

city = pd.Series(nasel, index=spis)

data['Nasel'] = data['City'].apply(lambda x: city[x])

rest_col=data.City.value_counts()

Nas_Res=[]

for i in data.City.keys():

    Nas_Res.append(data.Nasel[i]/rest_col[data.City[i]])

data['Nas_Res'] = pd.Series(Nas_Res)
data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: 1 if type(x)==float else len(x.split(',')))
# Ваша обработка 'Price Range'

price=[]

for i in range(len(data['Price Range'])):

    if type(data['Price Range'][i]) == float:

        price.append(0.5)

    elif len(data['Price Range'][i]) == 1:

        price.append(1)

    elif len(data['Price Range'][i]) == 4:

        price.append(4)

    else:

        price.append(2.5)

data['Price Range']=pd.Series(price)
# тут ваш код на обработку других признаков

data['Reviews']=data['Reviews'].fillna('')

rev = data['Reviews'].str[1:-1].str.split('[')

d_rev = []

for i in rev.keys():

    if len(pattern.findall(rev[i][-1]))==6:

        t = pattern.findall(rev[i][-1])

        date2 = t[-3:]

        date1 = t[-6:-3]

        tn =datetime.datetime.now() 

        t1 =datetime.datetime(int(date1[-1]),int(date1[-3]),int(date1[-2]))

        t2 =datetime.datetime(int(date2[-1]),int(date2[-3]),int(date2[-2]))

        if (tn-t1).days<(tn-t2).days:

            d_rev.append((tn-t1).days)

        else:

            d_rev.append((tn-t2).days)

    elif 0<len(pattern.findall(rev[i][-1]))<6:

        t = pattern.findall(rev[i][-1])

        date1 = t[-3:]

        tn =datetime.datetime.now() 

        t1 =datetime.datetime(int(date1[-1]),int(date1[-3]),int(date1[-2]))

        d_rev.append((tn-t1).days)

    else:

        d_rev.append(0)

d_rev=pd.Series(d_rev)

data['Reviews']=d_rev



data=pd.get_dummies(data, columns=['City'], dummy_na=True)

data=pd.get_dummies(data, columns=['Price Range'], dummy_na=True)

data.head(3)

data = data.drop(['Restaurant_id','URL_TA','ID_TA'], axis = 1)
data.info()
# Теперь выделим тестовую часть

train_data = data.query('sample == 1').drop(['sample'], axis=1)

test_data = data.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
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
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)