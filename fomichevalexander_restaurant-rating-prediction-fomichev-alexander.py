# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt

import seaborn as sns 

import re

import datetime

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split

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

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')



df_train.info()
df_train.head()
df_test.info()
df_test.head()
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями
df = df_test.append(df_train, sort=False).reset_index(drop=True)
df
df['Price Range'].value_counts()
df.isna().sum()
#Средний ценовой сегмент составляют большинство ресторанов. 

#Что ж, к пропущенным значениям можно также присвоить это значение:

df['Price Range']=df['Price Range'].fillna('$$ - $$$')
df['Price Range'].value_counts()
#Создадим функцию, конвертирующую символ доллара в численное значение:

def dollar_to_number(x):

    if x =='$':

        return 1

    if x =='$$ - $$$':

        return 2

    if x =='$$$$':

        return 3

dollar_to_number('$')
df['Price Range']=df['Price Range'].apply(dollar_to_number)
df
#Сколько городов представлено в наборе данных?

df.City.nunique()
df.City.value_counts()
not_capital = ["Krakow", "Lyon","Zurich","Hamburg","Barcelona","Oporto","Munich","Milan","Geneva"]

restaurant_in_capital=[]

restaurant_in_other_city=[]

for i in df['City']:

    if i in not_capital:

        restaurant_in_other_city.append(i)

    else:

        restaurant_in_capital.append(i)

        

print('Количество ресторанов в столицах:{}'.format(len(restaurant_in_capital)))

print('Количество ресторанов в других городах:{}'.format(len(restaurant_in_other_city)))



#Создадим функцию, которая численно выражает, находится ресторан в столице или нет:

not_capital = ["Krakow", "Lyon","Zurich","Hamburg","Barcelona","Oporto","Munich","Milan","Geneva"]

def Restaurant_in_capital(x):

    if x in not_capital:

        return 0

    return 1



Restaurant_in_capital('Paris')
df['Restaurant_in_capital']=df['City'].apply(Restaurant_in_capital)
df
#Сколько типов кухонь представлено в наборе данных?

df['Cuisine Style'].value_counts()
a = df['Cuisine Style'].str.split(', ')

kuhnya_list=[]

for b in range(len(df['Cuisine Style'])):

    if type(a[b])==float:

        continue

    for i in a[b]:

        kuhnya_list.append(i.replace('[','').replace(']','').replace("'",""))

#получится список из 119676 элементов, включающий все типы кухонь с каждого из 50000 ресторанов

print(len(kuhnya_list))

print(kuhnya_list)

#Чтобы удалить дубликаты, создадим словарь, используя ключи из списка, а затем создадим список на основе этого словаря:

kuhnya_dict = list(dict.fromkeys(kuhnya_list))

print(kuhnya_dict)

#Готово! Теперь дубликаты удалены. Проверим, сколько же типов кухонь содержит наш датасет:

print(len(kuhnya_dict))

#Чтобы посчитать, в скольких ресторанах содержится определённый тип кухни, импортируем Counter:

from collections import Counter

c = dict(Counter(kuhnya_list))

print(c)
restaurant_with_1_couisine_style=0

restaurant_with_2_couisine_style=0

restaurant_with_3_couisine_style=0

restaurant_with_4_couisine_style=0

restaurant_with_5_couisine_style=0

restaurant_with_6_couisine_style=0

restaurant_with_7_couisine_style=0

restaurant_with_8_couisine_style=0

restaurant_with_9_couisine_style=0

restaurant_with_10_couisine_style=0

restaurant_with_11_couisine_style=0

restaurant_with_13_couisine_style=0

restaurant_with_21_couisine_style=0

for i in range(0,len(df['Cuisine Style'])):

    if type(df['Cuisine Style'][i]) == float:

        continue

    elif len(df['Cuisine Style'][i].split(',')) == 1:

        restaurant_with_1_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 2:

        restaurant_with_2_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 3:

        restaurant_with_3_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 4:

        restaurant_with_4_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 5:

        restaurant_with_5_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 6:  

        restaurant_with_6_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 7:

        restaurant_with_7_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 8:

        restaurant_with_8_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 9:  

        restaurant_with_9_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 10:

        restaurant_with_10_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 11:

        restaurant_with_11_couisine_style+=1

    elif len(df['Cuisine Style'][i].split(',')) == 13:

        restaurant_with_13_couisine_style+=1

        print(i)

    elif len(df['Cuisine Style'][i].split(',')) == 21:

        restaurant_with_21_couisine_style+=1

        

print('Количество ресторанов с 1 типом кухни: {}'.format(restaurant_with_1_couisine_style))

print('Количество ресторанов с 2 типами кухни: {}'.format(restaurant_with_2_couisine_style))

print('Количество ресторанов с 3 типами кухни: {}'.format(restaurant_with_3_couisine_style))      

print('Количество ресторанов с 4 типами кухни: {}'.format(restaurant_with_4_couisine_style))

print('Количество ресторанов с 5 типами кухни: {}'.format(restaurant_with_5_couisine_style))

print('Количество ресторанов с 6 типами кухни: {}'.format(restaurant_with_6_couisine_style))

print('Количество ресторанов с 7 типами кухни: {}'.format(restaurant_with_7_couisine_style))

print('Количество ресторанов с 8 типами кухни: {}'.format(restaurant_with_8_couisine_style))

print('Количество ресторанов с 9 типами кухни: {}'.format(restaurant_with_9_couisine_style))

print('Количество ресторанов с 10 типами кухни: {}'.format(restaurant_with_10_couisine_style))

print('Количество ресторанов с 11 типами кухни: {}'.format(restaurant_with_11_couisine_style))

print('Количество ресторанов с 13 типами кухни: {}'.format(restaurant_with_13_couisine_style))

print('Количество ресторанов с 21 типом кухни: {}'.format(restaurant_with_21_couisine_style))

summa = restaurant_with_1_couisine_style+restaurant_with_2_couisine_style+restaurant_with_3_couisine_style+restaurant_with_4_couisine_style+restaurant_with_5_couisine_style+restaurant_with_6_couisine_style+restaurant_with_7_couisine_style+restaurant_with_8_couisine_style+restaurant_with_9_couisine_style+restaurant_with_10_couisine_style+restaurant_with_11_couisine_style+restaurant_with_13_couisine_style+restaurant_with_21_couisine_style

print('Сумма ресторанов, у которых имеется хотя бы один тип кухни: {}'.format(summa))

print('Количество ресторанов, у которых в графе Cuisine Style стоит NaN: {}'.format(df['Cuisine Style'].isna().sum()))

total = summa + df['Cuisine Style'].isna().sum()

print('Всего ресторанов: {}'.format(total))

#Какое среднее количество кухонь предлагается в одном ресторане? 

#Если в данных отсутствует информация о типах кухонь, то считайте, что в этом ресторане предлагается только один тип кухни. 

#Ответ округлите до одного знака после запятой.
a = df['Cuisine Style'].isna().sum()*1 + restaurant_with_1_couisine_style*1+restaurant_with_2_couisine_style*2+restaurant_with_3_couisine_style*3+restaurant_with_4_couisine_style*4+restaurant_with_5_couisine_style*5+restaurant_with_6_couisine_style*6+restaurant_with_7_couisine_style*7+restaurant_with_8_couisine_style*8+restaurant_with_9_couisine_style*9+restaurant_with_10_couisine_style*10+restaurant_with_11_couisine_style*11+restaurant_with_13_couisine_style*13+restaurant_with_21_couisine_style*21

print(a-11590)

print(a/total)
df['Cuisine Style'].str.split(',')
df['Cuisine Style'] = df['Cuisine Style'].fillna('Mix')
df['Cuisine Style'].isna().sum()
def number_of_types_of_kuhnya (x):

        return len(str(x).replace('[','').replace(']','').replace("'","").split(', '))

number_of_types_of_kuhnya (['Japanese', 'Sushi'])
df['Number_of_kuhnya'] = df['Cuisine Style'].apply(number_of_types_of_kuhnya)
df
df['URL_TA_number'] = df['URL_TA'].apply(lambda x: float(x[20:26]))
df['URL_TA_number'].value_counts()
df['ID_TA_number'] = df['ID_TA'].apply(lambda x: float(x[1:]))
df
# Заполним пропуски в количестве отзывов: если NaN - следовательно отзывов нет

df['Number of Reviews'] = df['Number of Reviews'].fillna(0)
population = {'Paris': 2190327, 'Stockholm': 961609, 'London': 8908081, 'Berlin': 3644826, 'Munich': 1456039, 'Oporto': 237591,'Milan': 1378689,'Bratislava': 432864, 'Vienna': 1821582, 'Rome': 4355725, 'Barcelona': 1620343, 'Madrid': 3223334,'Dublin': 1173179,'Brussels': 179277, 'Zurich': 428737, 'Warsaw': 1758143, 'Budapest': 1752286, 'Copenhagen': 615993,'Amsterdam': 857713,'Lyon': 506615, 'Hamburg': 1841179,'Lisbon': 505526, 'Prague': 1301132, 'Oslo': 673469,'Helsinki': 643272,'Edinburgh': 488100,'Geneva': 200548, 'Ljubljana': 284355,'Athens': 664046, 'Luxembourg': 115227,'Krakow': 769498}

df['Population'] = df['City'].map(population)
df
#Обычным срезом извлечь дату отзыва в каждом ресторане не получается

#Попробуем через регулярные выражения:
df['Reviews'] = df['Reviews'].fillna("['nodata']")
pattern = re.compile('\d+\/\d+\/\d+')

secret = pattern.findall("[['Better than expected.', 'Desperate, late night food.'], ['12/22/2017', '10/03/2017']]")

secret
df['Reviews_data'] = df['Reviews'].str.findall('\d+\/\d+\/\d+')
df['Reviews_data'][36]
def time_between_reviews(row):

    if row['Reviews_data'] == []:

         return None

    return pd.to_datetime(row['Reviews_data']).max() - pd.to_datetime(row['Reviews_data']).min()

    
df['Time_between_reviews'] = df.apply(time_between_reviews, axis=1)
df['Time_between_reviews'] = df['Time_between_reviews'].apply(lambda x: x.days) 
df['Time_between_reviews'].median()
df['Time_between_reviews'].max()
df['Time_between_reviews'] = df['Time_between_reviews'].fillna(df['Time_between_reviews'].median())
df['Reviews_data'][df['Time_between_reviews']==3296]
def time_fromlastreview_to_today(row):

    if row['Reviews_data'] == []:

         return None

    return pd.datetime.now() - pd.to_datetime(row['Reviews_data']).max()
df['Time_from_last_review_to_today'] = df.apply(time_fromlastreview_to_today, axis=1)
df['Time_from_last_review_to_today'] = df['Time_from_last_review_to_today'].dt.days
df['Time_from_last_review_to_today']
df['Time_from_last_review_to_today'].mean()
df['Time_from_last_review_to_today'] = df['Time_from_last_review_to_today'].fillna(df['Time_from_last_review_to_today'].mean())
df['Time_from_last_review_to_today'] = df['Time_from_last_review_to_today'].round()
# Для создания dummy variables обратим внимание на столбцы City и Price Range:
a = pd.get_dummies(df, columns=['City'])   

a
df_finish = pd.get_dummies(a, columns=['Price Range'])

df_finish
df_finish=df_finish.drop(['Cuisine Style','Reviews','URL_TA','ID_TA','Reviews_data','Restaurant_id'], axis=1)
# Теперь выделим тестовую часть

train_data = df_finish.query('sample == 1').drop(['sample'], axis=1)

test_data = df_finish.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
#Проверяем

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



test_data.info()
test_data = test_data.drop(['Rating'], axis=1)
test_data
sample_submission
predict_submission = model.predict(test_data)
predict_submission
# Округлим результаты работы модели:

def round_to_polovina(row):

    return (round(row*2.0)/2)



new_round = np.vectorize(round_to_polovina)

y_pred_round = new_round(model.predict(X_test))

print('MAE:', metrics.mean_absolute_error(y_test, y_pred_round))
sample_submission['Rating'] = predict_submission

sample_submission['Rating'] = sample_submission['Rating'].apply(round_to_polovina)

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)