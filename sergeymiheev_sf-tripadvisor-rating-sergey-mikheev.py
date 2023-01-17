# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter# счетчик



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
col = ['Restaurant_id', 'City', 'Cuisine_Style', 'Ranking', 'Price_Range', 'Number_of_Reviews', 'Reviews', 'URL_TA', 'ID_TA','sample','Rating']

data.columns = col
data.info(100000)
data
data['Restaurant_id'].value_counts()

# пропусков нет, все ок
print(data['Restaurant_id'].describe())

# по условию - в одному id  могут быть прикреплены несколько ресторанов одной сети,

# поэтому уникальных значений меньше чем записей
data['Restaurant_id'] = [i[3:] for i in data['Restaurant_id']]

data['Restaurant_id'] = [int(i) for i in data['Restaurant_id']]



print(' ')

print(len(data['Restaurant_id']))

print(type(data['Restaurant_id'][1]))



# превратили столбец в численное выражение
a=data['Restaurant_id'].value_counts()

a=dict(a)

a=pd.DataFrame({'count':a}, columns=['count'])



A=a.index

B=a.values



data['chain']= data['Restaurant_id'].replace(A, B)

data[['Restaurant_id', 'chain']]
data.City.value_counts()

# пропусков нет, города не повторяются, проблем с ракладкой или регистром нет, все ок


print(data.City.describe())
#преобразуем City в переменную dummy

data = pd.concat((data, pd.get_dummies(data.City)), axis=1)

data.info()
A = list(data.City.value_counts().keys())

data1=pd.DataFrame()

data1['City'] = data['City']

data1['Ranking']=data['Ranking']

data1=pd.DataFrame(data1.groupby(['City']).max())

data2=data1['Ranking']

A=list(data2.keys())

B=list(data2)



data['Len_rest_in_city'] = data['City'].replace(A, B)



data['Std_ranking']=data['Ranking']/data['Len_rest_in_city']



data.Std_ranking.describe()
data2
number_city_residents_dict = {'Paris':2148327, 'Stockholm':961609, 'London':8908081, 'Berlin':3644826, 'Munich':1471508, 'Oporto':237591,

       'Milan':1378689, 'Bratislava':437725, 'Vienna':1897491, 'Rome':2870500, 'Barcelona':1636762, 'Madrid':3266126,

       'Dublin':1173179, 'Brussels':179277, 'Zurich':428737, 'Warsaw':1790658, 'Budapest':1752286, 'Copenhagen':615993,

       'Amsterdam':872757, 'Lyon':506615, 'Hamburg':1841179, 'Lisbon':505526, 'Prague':1301132, 'Oslo':673469,

       'Helsinki':655281, 'Edinburgh':488100, 'Geneva':200548, 'Ljubljana':284355, 'Athens':664046,

       'Luxembourg':602005, 'Krakow':769498}

# словарь со кол-вом жителей

data['population'] = data['City'].replace(number_city_residents_dict)

print(len(data.population))

print(type(data.population[1]))
country_dict = {'London': 'GreatBritain', 'Paris': 'France', 'Madrid': 'Spain', 'Barcelona': 'Spain', 

                'Berlin': 'Germany', 'Milan': 'Italy', 'Rome': 'Italy', 'Prague': 'CzechRepublic', 

                'Lisbon': 'Portugal', 'Vienna': 'Austria', 'Amsterdam': 'Netherlands', 'Brussels': 'Belgium', 

                'Hamburg': 'Germany', 'Munich': 'Germany', 'Lyon': 'France', 'Stockholm': 'Sweden', 

                'Budapest': 'Hungary', 'Warsaw': 'Poland', 'Dublin': 'Irland', 'Copenhagen': 'Denmark', 

                'Athens': 'Greece', 'Edinburgh': 'Scotland', 'Zurich': 'Switzeland', 'Oporto': 'Portugal', 

                'Geneva': 'Switzeland', 'Krakow': 'Poland', 'Oslo': 'Norway', 'Helsinki': 'Finland', 

                'Bratislava': 'Slovakia', 'Luxembourg': 'Luxembourg', 'Ljubljana': 'Slovenia'

}

# словарь города-страны



data['Country'] = data.City.replace(country_dict)

print(data.Country.value_counts())



A = list(data.Country.value_counts().keys())

B = range(0, len(A))

dict_country = dict(zip(A, B))

# словарь со значениями стран





data['Country_ind'] = data['Country'].replace(A, B)



print(len(data.Country_ind))

print(type(data.Country_ind[1]))
data['rew_on_pop'] = data['Number_of_Reviews']/data['population']

data['rew_on_pop'] = data['rew_on_pop'].fillna(0)



print(len(data.rew_on_pop))

print(type(data.rew_on_pop[1]))
# # сформируем список достаточно уникальных кухонь и сформируем на его основе новый признак

# list_of_unique_Cuisine = [x[0] for x in temp_counter.most_common()[-16:]]

# data['unique_Cuisine_Style'] = data['Cuisine_Style'].apply(lambda x: 1 if len(set(x) & set(list_of_unique_Cuisine))>0  else 0).astype('float64')
len(data[data['Cuisine_Style'].isna()])
# в переменной 11590 (23.2%) пропущенных значений 

# сохраним эту информацию

data['NAN_Cuisine Style'] = pd.isna(data['Cuisine_Style']).astype('float64') 





data['Cuisine_Style'] = data['Cuisine_Style'].fillna("['No_info']")

# заменяем пропуски



# закодируем значения в переменной до их преобразования

le = LabelEncoder()

le.fit(data['Cuisine_Style'])

data['code_Cuisine Style'] = le.transform(data['Cuisine_Style'])
# проведем обработку значений переменной

data['Cuisine_Style'] = data['Cuisine_Style'].str.findall(r"'(\b.*?\b)'")



temp_list = data['Cuisine_Style'].tolist()



def list_unrar(list_of_lists):

    # функция создает общий список стилей кухни

    result=[]

    for lst in list_of_lists:

        result.extend(lst)

    return result



# создаем счетчик стилей кухни

coun=Counter(list_unrar(temp_list))







# сформируем список достаточно уникальных кухонь и сформируем на его основе новый признак

list_of_unique_Cuisine = [x[0] for x in coun.most_common()[-16:]]

data['unique_Cuisine_Style'] = data['Cuisine_Style'].apply(lambda x: 1 if len(set(x) & set(list_of_unique_Cuisine))>0  else 0).astype('float64')



data['Cuisine_Style2'] = data['Cuisine_Style']



# coun = Counter(b)  # подсчитываем количество каждого стиля кухни

coun = dict(coun)

coun = pd.DataFrame({'count': coun}, columns=['count'])

# находим среднее количество используемых стилей кухни

a = coun['count'].mean()



b = set(coun.query('count > @a').index)

# оставляем только самые популярные кухни



c = set(coun.index) - b

# находим непопулярные кухни



cus_st = pd.DataFrame(data['Cuisine_Style2'].to_list()).stack()

for item in c:

    cus_st = cus_st.replace(item, 'other_cuisine_style')

cus_st = cus_st.groupby(level=[0]).apply(",".join).reset_index()

data['Cuisine_Style'] = cus_st[0]

# заменяем непопулярные кухни на 'other'





b.add('other_cuisine_style')



def find_item(cell):

    if item in cell:

        return 1

    return 0





for item in b:

    data[item] = data['Cuisine_Style'].apply(find_item).astype(np.uint8)

# преобразуем в dummy переменную





data['count_Cuisine_Style'] = data['Cuisine_Style2'].apply(lambda x: len(x))

# подсчитываем количество используемых стилей кухни в ресторане



print(len(data['Cuisine_Style']))

print(len(data['count_Cuisine_Style']))

# пропуски заменены, строки преобразованы в столбцы
data.info()
print(data.Price_Range.describe())
price_dict ={'$':1, '$$ - $$$':2, '$$$$':3}

data['price_rank'] = data['Price_Range'].replace(price_dict)

# делаем замену текста на числа



price_mode = data['price_rank'].mode()

data['price_rank'].fillna(int(price_mode), inplace=True)

# пропущенные значения заменяем самой частой величиной



print('')

print(len(data.price_rank))

print(type(data.price_rank[1]))

# заменили буквенные индефикаторы на численные
print(data.Reviews.describe())
from datetime import datetime, date, time

data['Reviews'] = data.Reviews.replace("[[], []]", 'No_info')

# заменяем пропуски
data['Last_rew'] = data['Reviews']



data['Last_rew']=data['Last_rew'].str[-27:-17]



now = datetime.now()
data['Last_rew']
#base['Last_rew'][base.Last_rew.str.contains("]")]=now

data['Last_rew'][data.Last_rew.str.contains("]")==True] = '01/01/2000'

data['Last_rew'] = data['Last_rew'].fillna('01/01/2000')
data['Last_rew'] = data['Last_rew'].apply(pd.to_datetime) 
data['Last_rew_data'] = now - data['Last_rew']
data['Last_rew_data'] = data['Last_rew_data'].apply(lambda x: x.total_seconds())# [i.total_seconds() for i in data.Last_rew_data]
data['Last_rew_data'] = data['Last_rew_data'].fillna(data['Last_rew_data'].max())

data['Last_rew_data'] = data['Last_rew_data'].astype(np.int64)

print('')

print(len(data.Last_rew_data))

print(type(data.Last_rew_data[1]))
m=data['Number_of_Reviews'].mean()

print(data['Number_of_Reviews'].describe())

# отрицательных чисел нет


data['Number_of_Reviews'].value_counts()

# но есть пропуски, нужно проверить, что где есть пропуски - там действительно нет отзывов
data['Number_of_Reviews'] = data['Number_of_Reviews'].fillna('No_info')

# меняем пропуски на Not_info
a = data.query('Reviews == "No_info" & Number_of_Reviews == "No_info"')

b = list(a.index)

data['Number_of_Reviews'][b] = 0

# где нет значения кол-ва отзывов и превью отзывов - ставим 0
len(data['Number_of_Reviews'])

# заменили предполагаемые пропуски
data['Number_of_Reviews'] = data['Number_of_Reviews'].replace('No_info', m)



# пропуски, которые не смогли заполнить, заполняем медианным значением



print('')

print(len(data.Number_of_Reviews))

print(type(data.Number_of_Reviews[1]))
A = list(data.City.value_counts().keys())

data1=pd.DataFrame()

data1['City'] = data['City']

data1['Number_of_Reviews']=data['Number_of_Reviews']
data2=pd.DataFrame(data1.groupby(['City']).sum())

data2=data2['Number_of_Reviews']

A=list(data2.keys())

B=list(data2)

data['Len_rew'] = data['City'].replace(A, B)



data['Std_num_rew']=data['Number_of_Reviews']/data['Len_rew']
print(data['ID_TA'].describe())
data['ID_TA']=data['ID_TA'].str[1:]

data.ID_TA = [float(i) for i in data.ID_TA]
print('')

print(len(data.ID_TA))

print(type(data.ID_TA[1]))
plt.rcParams['figure.figsize'] = (10,7)

data['Ranking'].hist(bins=100)
data['City'].value_counts(ascending=True).plot(kind='barh')
data['Ranking'][data['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (data['City'].value_counts())[0:10].index:

    data['Ranking'][data['City'] == x].hist(bins=100)

plt.show()
data['Rating'].value_counts(ascending=True).plot(kind='barh')
data['Ranking'][data['Rating'] == 5].hist(bins=100)
data['Ranking'][data['Rating'] < 4].hist(bins=100)
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data[col].drop(['sample'], axis=1).corr(),)
data = data.drop('URL_TA', axis=1)

data = data.drop('Reviews', axis=1)

data = data.drop('Last_rew', axis=1)

data = data.drop('City', axis=1)

data = data.drop('Country', axis=1)

data = data.drop('Restaurant_id', axis=1)

data = data.drop('chain', axis=1)

data = data.drop('Cuisine_Style', axis=1)

data = data.drop('Cuisine_Style2', axis=1)

data = data.drop('Price_Range', axis=1)

data.info()
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)


data.corr()
pd.options.display.max_info_columns 

data.info(verbose=True, max_cols=False, null_counts=True)
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
len(test_data)
len(sample_submission)
predict_submission = model.predict(test_data)



len(predict_submission)
def round_nearest(x, a):

    return round(x / a) * a



sample_submission['Rating'] = predict_submission.round(1)

sample_submission['Rating'] = round_nearest(sample_submission['Rating'], 0.5)





sample_submission.head(10)



sample_submission.to_csv('submission.csv', index=False)