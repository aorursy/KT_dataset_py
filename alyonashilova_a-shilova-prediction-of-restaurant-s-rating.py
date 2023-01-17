# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#фиксируем RANDOM_SEED

RANDOM_SEED = 42
#фиксируем тип пакетов

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
df_train.head(5)
df_test.info()
df_test.head()
sample_submission.head()
sample_submission.info()
# Объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
mean_num_of_rev = round(data['Number of Reviews'].mean(),1)

data['Number of Reviews']= data['Number of Reviews'].fillna(mean_num_of_rev)
data.head()
data['Price Range'].value_counts()
#Обозначим ценовые диапазоны соответственно числами 1(самый низкий),2(средний),3(высокий). 

#Nan заменим на 2, как на медианное значение

price_range = {'$':1,'$$ - $$$':2,'$$$$':3,float('nan'):2}

data['Price Range'] = data['Price Range'].map(price_range)
data.head()
capitals ={'London':1, 'Paris':1, 'Madrid':1, 'Barcelona':0, 'Berlin':1,

      'Milan':0, 'Rome':1, 'Prague':1, 'Lisbon':1,

      'Vienna':1, 'Amsterdam':1, 'Brussels':1, 'Hamburg':0,

      'Munich':0, 'Lyon':0, 'Stockholm':1, 'Budapest':1,

      'Warsaw':1, 'Dublin':1, 'Copenhagen':1, 'Athens':1,

      'Edinburgh':1, 'Zurich':0, 'Oporto':0, 'Geneva':0,

      'Krakow':0, 'Oslo':1, 'Helsinki':1, 'Bratislava':1,

      'Luxembourg':1, 'Ljubljana':1}

data['Capitals'] = data['City'].map(capitals)
data.head()
city=data['City'].value_counts()

norm_rank=[]

for i in data.index:

    norm_rank.append(round((data['Ranking'][i]/city[data['City'][i]]),2))

data['Norm_Ranking']= norm_rank
data.head()
#Заполняем отсутствующие значения

data['Cuisine Style'] = data['Cuisine Style'].fillna('["none"]')

#Вместо 'none' добавляем 1, потому что так рекомендовали в учебных материалах

C_Style=[]

for item in data['Cuisine Style']:

    if item=='["none"]':

        C_Style.append(1)

    else:

        C_Style.append(len(item[2:-2].split("', '")))

data['Count_CS']= C_Style
data.head()
#Создаём серию, в которой содержатся списки кухонь каждого ресторана

Cuisine_Style = data['Cuisine Style'].str[2:-2].str.split("', '")

#Cuisine_Style.head()
#создаём общий список кухонь всех ресторанов

CS_All=[]   

for item in Cuisine_Style:

    for i in range(len(item)):

        CS_All.append(item[i])

CS_All[:5]
#Находим 10 самых популярных видов кухни

Сousine_Style_All=pd.Series(CS_All)

Most_Pop_Cuis=Сousine_Style_All.value_counts().head(11).drop(labels = ['none'])

Most_Pop_Cuis
#создаём датафрейм из популярных кухонь

Cuisine_Style_Pop=pd.DataFrame(columns=Most_Pop_Cuis.index, index=data.index)       

Cuisine_Style_Pop.head()
#заполняем его

for cuisine in Most_Pop_Cuis.index:

    for i in range(len(Cuisine_Style_Pop)):

        if cuisine in CS_All[i]:

            Cuisine_Style_Pop[cuisine][i]=1

        else:

            Cuisine_Style_Pop[cuisine][i]=0
# к сожалению, не заполнила отсутствующие значения самыми популярными кухнями по городу, осознаю этот недостаток

Cuisine_Style_Pop.head(10)
#добавим словарь с количеством населения из википедии

city_pop ={'London':8908100, 'Paris':2190300, 'Madrid':3165500, 'Barcelona':1636700, 'Berlin':3644800,

      'Milan':1378700, 'Rome':2875800, 'Prague':1301100, 'Lisbon':505500,

      'Vienna':1897500, 'Amsterdam':1857700, 'Brussels':179300, 'Hamburg':1841200,

      'Munich':1471500, 'Lyon':506600, 'Stockholm':961600, 'Budapest':1752300,

      'Warsaw':1783300, 'Dublin':1173200, 'Copenhagen':616000, 'Athens':664000,

      'Edinburgh':488100, 'Zurich':428700, 'Oporto':237600, 'Geneva':200500,

      'Krakow':769500, 'Oslo':673500, 'Helsinki':643300, 'Bratislava':425900,

      'Luxembourg':115200, 'Ljubljana':284400}

#Создадим список с количеством населения в городах, где находятся рестораны

rest_PC=[]

for i in data['City']:

    rest_PC.append(city_pop[i])
#соберём список количества ресторанов по городам

num_RC=[]

for item in data['City']:

    num_RC.append(city[item])
#посчитаем отношение количества ресторанов к населению городов

Rel_pop_rest=pd.DataFrame(data=rest_PC, columns = ['City_Pop'])

Rel_pop_rest['Rest_Count']=num_RC

Rel_pop_rest['RPR'] = round((Rel_pop_rest['City_Pop']/Rel_pop_rest['Rest_Count']),1)

Rel_pop_rest.head()
#добавим полученный результат к исследуемому датафрейму

data['Rel_Pop_Rest']=Rel_pop_rest['RPR']

data.head()
data[['Vegetarian Friendly','European', 'Mediterranean',

    'Italian','Vegan Options','Gluten Free Options','Bar','French','Asian','Pizza']] = Cuisine_Style_Pop

data.head()
data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.head()
data.drop(['Restaurant_id','Cuisine Style','Reviews','URL_TA','ID_TA',], axis = 1, inplace=True)
data.head()
train_data = data.query('sample == 1').drop(['sample'], axis=1)

test_data = data.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)

# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)