# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import re



dfY=pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/kaggle_task.csv')

dfX=pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/main_task.csv')



#дополнителняем dfY на Rating, добавляем признак выборки

dfY['Rating']=0

dfY['Type']=False

dfX['Type']=True



#склеиваем

df=pd.concat([dfX,dfY])





# генерим переменные как кол-во пропусков в соответствующих столбцах

df['na_cuis']=pd.isna(df['Cuisine Style']).astype('int')

df['na_nor']=pd.isna(df['Number of Reviews']).astype('int')

df['na_por']=pd.isna(df['Price Range']).astype('int')



# присваиваем пропуск в ['Price Range'] добавляем числовое поле с уровнем цен

df['Price Range'].fillna('$',inplace=True)

price_dict = {'$$ - $$$':2.5, '$$$$':4, '$':1}

df['PR']=df['Price Range'].map(price_dict)



# считаем кол-во кухонь

df['Cuisine Style'].fillna("['European']",inplace=True)

df['count_cuis']=df['Cuisine Style'].str[2:-2].str.split("', '").dropna(0).apply(lambda x: len(x))



# добавление фич срдней стоимости по городу и относительной стоимости по городу

avcityprice=df.groupby(['City'])['PR'].mean()

df['avPriceCity']=df['City'].map(lambda x:avcityprice[x])

df['kPR']=df['PR']/df['City'].map(lambda x:avcityprice[x])



# добавление фич наибольшего рэнка по городу и относительного рэнка по городу

maxcityrank=df.groupby(['City'])['Ranking'].mean()

df['maxcityrank']=df['City'].map(lambda x:maxcityrank[x])

df['kcityRanking']=df['Ranking']/df['City'].map(lambda x:maxcityrank[x])



#отношение относительного рэнка по городу к относительной цене:

df['avRPice']=(df['kcityRanking'])/df['kPR']



# добавляем отношение удельной стоимости рэнка к количеству кухонь

df['kRPC']=(df['avRPice']/df['count_cuis'])



# добавляем признак столицы

ci=pd.DataFrame(df['City'].unique())

ci.columns=(['City'])

ci['stol']=ci['City'].apply(lambda x: 1 if x in['Paris','Edinburgh','Lisbon','London','Luxembourg','Stockholm','Athens','Berlin','Bratislava','Vienna','Ljubljana','Helsinki','Oslo','Prague','Rome','Madrid','Dublin','Brussels','Warsaw','Budapest','Copenhagen','Amsterdam'] else 0)

df=pd.merge(df,ci,on='City',how='inner')



#заполняем пропуски в количестве просмотров через средние по городу

meansNumberRewiews = df[df['Number of Reviews']>0].groupby('City')['Number of Reviews'].mean().round()

df['Number of Reviews'] = df.apply(lambda row: meansNumberRewiews[row['City']] if pd.isnull(row['Number of Reviews']) else row['Number of Reviews'], axis=1)



dfcityReview=df[df['Number of Reviews']>0].groupby(['City'])['Number of Reviews'].mean().round()

# добавляем фичу отношения кол-ва отзывов к срднегородскому кол-ву отзывов

df['avNofR']=df['Number of Reviews']/df['City'].map(dfcityReview)



# добавляем фичу отношения относительного кол-ва отзывов к относительной цене по городу

df['avNofR_vs_avRPice']=df['avNofR']/df['avRPice']



# генерим dummy по городу

df= pd.get_dummies(df, prefix='City_', columns=['City'])



# очистка от текстовых столбцов

df=df.drop(['Reviews','URL_TA','ID_TA','Cuisine Style','Price Range'],axis=1)

train_df = df[df['Type']]

# Разбиваем датафрейм на части, необходимые для обучения и тестирования модели

X = train_df.drop(['Restaurant_id', 'Rating'], axis = 1)

y = train_df['Rating']



# Загружаем специальный инструмент для разбивки:

from sklearn.model_selection import train_test_split



# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.

# Для тестирования мы будем использовать 1% от исходного датасета.



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)



# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели



# Создаём модель

regr = RandomForestRegressor(n_estimators=100)



# Обучаем модель на тестовом наборе данных

regr.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = (2*regr.predict(X_test)).round(0)/2



# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

valid_df = df[~df['Type']].drop(['Restaurant_id', 'Rating'],axis=1)

#valid_df.info()

valid_y_pred = (2*regr.predict(valid_df)).round(0)/2

# Создадим датасет конечного результата submission_df

submission_df = pd.DataFrame()

# Запишем в него требуемые данные

submission_df['Restaurant_id'] = df[~df['Type']]['Restaurant_id']

submission_df['Rating'] = valid_y_pred



# Сохраним результат в файл

submission_df.to_csv('solution.csv', index=False)
