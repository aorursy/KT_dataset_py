# Импортируем необходимые библиотеки:
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import re
import requests
from bs4 import BeautifulSoup
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Для воспроизводимости результатов:
random_seed = 42 # Общий параметр для генерации случайных чисел
current_date = pd.to_datetime('18/04/2020') # Общая текущуя дата
!pip freeze > requirements.txt # Для тех, кто будет запускать мой ноутбук на своей машине создаем requirements.txt
# Загрузим датасеты
mt_data = pd.read_csv('../input/sf-dst-restaurant-rating/main_task.csv')
kt_data = pd.read_csv('../input/sf-dst-restaurant-rating/kaggle_task.csv')
mt_data.head(3)
mt_data.info()
kt_data.head(3)
kt_data.info()
# Рассмотрим последние два стролбца ДФ.
print('URL_TA[0]  ', mt_data['URL_TA'][0], '\n\n', 'ID_TA[0]  ', mt_data['ID_TA'][0], sep = '')

# URL_TA очевидно ссылка на ресторан на Tripadvisor
# ID_TA судя по всему - id ресторана в системе Tripadvisor, так как присутствует в ссылке на ресторан

print('\nУникальных ID_TA', mt_data['ID_TA'].nunique())

# судя по всему, в данных есть задвоение ресторанов, найдем их

rest_id_ta = mt_data['ID_TA'].value_counts(sort = True, ascending = False)
print('\nУникальные значения rest_id_ta', rest_id_ta.unique())
print('\nКоличество задвоений', len(rest_id_ta.loc[rest_id_ta == 2]))
# Как видно, 20 ресторанов представлены в ДФ дважды, убедимся что это именно одинаковые строки
mt_data[(mt_data['ID_TA'] == 'd2477531') | (mt_data['ID_TA'] == 'd7342803') | (mt_data['ID_TA'] == 'd697406')]

# Все не так однозначно - все колонки одинаковые, кроме 'Ranking' и 'Restaurant_id'
# учитывая столь большие сходства будем считать эти строки дубликатами и уберем лишние
# Рассмотрим последние два стролбца в ДФ kaggle_task.
print('URL_TA[0]  ', kt_data['URL_TA'][0], '\n\n', 'ID_TA[0]  ', kt_data['ID_TA'][0], sep = '')

# URL_TA очевидно ссылка на ресторан на Tripadvisor
# ID_TA судя по всему - id ресторана в системе Tripadvisor, так как присутствует в ссылке на ресторан

print('\nУникальных ID_TA', kt_data['ID_TA'].nunique())

# В этом ДФ задвоения ресторанов нет
# Для объединения будем брать среднее значение, округленное до единицы
# Заменяем Ранги задвоенных ресторанов на среднееарифметическое 
rest_ids = list(rest_id_ta.loc[rest_id_ta == 2].index)
for rest_id in rest_ids:
    ranking_mean = round(mt_data[mt_data['ID_TA'] == rest_id]['Ranking'].mean())
    mt_data.at[mt_data[mt_data['ID_TA'] == rest_id].index[0], 'Ranking'] = ranking_mean
    mt_data.at[mt_data[mt_data['ID_TA'] == rest_id].index[1], 'Ranking'] = ranking_mean

# Удаляем дубликаты
mt_data = mt_data.drop_duplicates(subset = ['ID_TA'])
mt_data.info()
# Объединим ДФ main и kaggle для проведения предобработки и обогащения данных.
# В ДФ kaggle добавим столбец 'Rating' со значением 0
# Для различения датафреймов добавим стобец 'Main'
mt_data['Main'] = 1
kt_data['Main'] = 0
kt_data['Rating'] = 0
df = mt_data.append(kt_data, sort = False).reset_index(drop = True)
df.info()
df.head(3)
# Предполагая, что свои 'ID_TA' Tripadvisor назначает в порядке добавления ресторана в систему,
# переведем параметр в числовой и будет использовать в расчетной модели
df['ID_TA'] = df['ID_TA'].apply(lambda x: int(x[1:]))
df.head(3)
# Количетсво уникальных городов в ДФ
df['City'].nunique()
# Количество уникальных id ресторанов в ДФ
df['Restaurant_id'].nunique()
# Оказывается есть рестораны оцененные несколько раз
df[df['Restaurant_id'] == 'id_1535']
# Видимо параметр 'Restaurant_id' не является общим для всей выборки так как нет повторяющихся городов,
# то есть параметр применяется для каждой страны отдельно, таким образом в ДФ представлены 40000 уникальных ресторанов
# Создаю словарь с количеством ресторанов в каждом городе
restaurant_count = {}
for city in df['City']:
    if city not in restaurant_count:
        restaurant_count[city] = 1
    else:
        restaurant_count[city] += 1
restaurant_count
df['Rest. Count'] = df['City'].map(restaurant_count)
df.head(3)
df['Relative Ranking'] = df['Ranking'] / df['Rest. Count']
df.head(3)
print(df['Price Range'].unique(), df['Price Range'].value_counts(), sep = '\n\n')
# В ДФ рестораны разбиты на 3 ценовых диапазона от самого дешевого '$' до самого дорогого '$$$$'.
# Неизвестные данные примем как среднюю ценовую группу '$$-$$$', так как он наиболее часто встречается
# и добавим столбец dummy-параметра о том, приведен ли параметр 'Price Range' в ДФ (False), или нет (True).
df['P.R. nan'] = df['Price Range'].isna()
df['Price Range'] = df['Price Range'].apply(lambda x: 1 if x == '$' else (5 if x == '$$$$' else 3))
df.head(3)
# Поскольку для всех ресторанов представлены отзывы (параметры URL_TA и ID_TA не имеют пропусков),
# то недостающие значения параметра 'Number of Reviews' заполним средним арифметическим с добавлением dummy-параметра
df['NoR nan'] = df['Number of Reviews'].isna()
df['Number of Reviews'] = df['Number of Reviews'].fillna(df['Number of Reviews'].mean())
df.head(3)
# Создадим словарь с данными о каждом городе, в котором собрана информация о:
# названии страны (анг.)[0], названии страны (рус.)[1], статусе столицы [2], населении города [3], количестве туристов [5].
# Названия стран и городов пригодятся в дальнейшем для парсинга данных.
# Население городов и количество туристов в стране изначально парсил по таблицам ООН и Международной туристической организации,
# но в таком случае очень долго обрабатыватся таблицы, так что добавил в этот словарь.

cities_data = {
    'Paris': ['France', 'Франция', True, 2206488, 89322],
    'Stockholm': ['Sweden', 'Швеция', True, 789024, 10522],
    'London': ['United Kingdom', 'Великобритания', True, 8135667, 36316],
    'Berlin': ['Germany', 'Германия', True, 3613495, 38881],
    'Munich': ['Germany', 'Германия', False, 1456039, 38881],
    'Oporto': ['Portugal', 'Португалия', False, 214353, 16186],
    'Milan': ['Italy', 'Италия', False, 1358871, 61567],
    'Bratislava': ['Slovakia', 'Словакия', True, 427744, 7620],
    'Vienna': ['Austria', 'Австрия', True, 1888776, 30816],
    'Rome': ['Italy', 'Италия', True, 2873147, 61567],
    'Barcelona': ['Spain', 'Испания', False, 1620576, 82773],
    'Madrid': ['Spain', 'Испания', True, 3203157, 82773],
    'Dublin': ['Ireland', 'Ирландия', True, 544107, 10926],
    'Brussels': ['Belgium', 'Бельгия', True, 174383, 9119],
    'Zurich': ['Switzerland', 'Швейцария', False, 415367, 11715],
    'Warsaw': ['Poland', 'Польша', True, 1754511, 19622],
    'Budapest': ['Hungary', 'Венгрия', True, 1751219, 17552],
    'Copenhagen': ['Denmark', 'Дания', True, 616098, 12749],
    'Amsterdam': ['Netherlands', 'Нидерланды', True, 821752, 18780],
    'Lyon': ['France', 'Франция', False, 513275, 89322],
    'Hamburg': ['Germany', 'Германия', False, 1830584, 38881],
    'Lisbon': ['Portugal', 'Португалия', True, 505526, 16186],
    'Prague': ['Czech Republic (Czechia)', 'Чехия', True, 1294513, 13665],
    'Oslo': ['Norway', 'Норвегия', True, 634293, 5688],
    'Helsinki': ['Finland', 'Финляндия', True, 639227, 3224],
    'Edinburgh': ['United Kingdom', 'Великобритания', False, 482005, 36316],
    'Geneva': ['Switzerland', 'Швейцария', False, 201818, 11715],
    'Ljubljana': ['Slovenia', 'Словения', True, 279650, 4425],
    'Athens': ['Greece', 'Греция', True, 664046, 30123],
    'Luxembourg': ['Luxembourg', 'Люксембург', True, 116323, 1018],
    'Krakow': ['Poland', 'Польша', False, 759104, 19622]
}
df['Capital'] = df['City'].apply(lambda x: cities_data[x][2])
df.head(3)
print(df['Cuisine Style'][0], type(df['Cuisine Style'][0]))
# Данные в столбце 'Cuisine Style' представлены в виде списка, который оформлен в строку
# Сначала заполним пустоты аналогичной строкой '[]', добавим dummy-параметр для пустых данных, приведем данные к типу list
df['CS nan'] = df['Cuisine Style'].isna()
df['Cuisine Style'] = df['Cuisine Style'].fillna('[]')
df['Cuisine Style'] = df['Cuisine Style'].apply(lambda x: eval(x))
print(df['Cuisine Style'][0], type(df['Cuisine Style'][0]))
df.head(3)
# Для пустых значений примеем 1
df['Cuisine Count'] = df['Cuisine Style'].apply(lambda x: len(x) if len(x) > 0 else 1)
df.head(3)
#Составляю список кухонь
cuisine_iter = tuple([set(x) for x in df['Cuisine Style']])
cuisine_set = sorted(set.union(*cuisine_iter))
len(cuisine_set)
# Считаю, сколько раз представлена в ДФ каждая кухня
cuisine_set_count = {}
for cuisine in cuisine_set:
    cuisine_set_count[cuisine] = 0
    for row in df['Cuisine Style']:
        if cuisine in row:
            cuisine_set_count[cuisine] += 1
cuisine_set_count = pd.DataFrame(data = [*cuisine_set_count.items()], columns = ['Cuisine', 'Count'])
cuisine_set_count.sort_values(by = ['Count'], ascending = False)
# Создаю список кухонь, представленных единожды в ДФ
unique_cuisine = list(cuisine_set_count[cuisine_set_count['Count'] == 1]['Cuisine'])
unique_cuisine
#Создаю функцию определения наличия уникальной кухни в ресторане
def Unique_Cuisine(data):
    result = 0
    for cuisine in unique_cuisine:
        if cuisine in data:
            result = 1
    return result
     
#Добавляю столбец параметов уникальных кухонь    
df['Unique Cuisine'] = df['Cuisine Style'].apply(lambda x: Unique_Cuisine(x))
df[df['Unique Cuisine'] == 1]
# Смотрим содержание в столбце 'Reviews'
df['Reviews'][0]
# Можно извлеч даты последнего и предпоследнего отзыва, определить разницу во времени между отзывами,
# разницу между последним отзывом и текущей датой
# Сперва заполним пропуски в данных
df['Reviews'] = df['Reviews'].fillna('[[], []]')
# Создаю отдельный датафрейм из дат отзывов и расчетом разницы между этими датами
pattern = re.compile('\d{2}/\d{2}/\d{4}')
reviews = []
for i in df['Reviews']:
    reviews.append(re.findall(pattern, i))
rev = pd.DataFrame(reviews)
rev.columns = ['date1', 'date2']
rev['date1'] = pd.to_datetime(rev['date1']) 
rev['date2'] = pd.to_datetime(rev['date2']) 
rev['TD Reviews'] = (rev['date1'] - rev['date2']).dt.days
rev['Last Review'] = (current_date - rev['date1']).dt.days
rev.info()
# Пропуски в столбцах 'Deta Reviews' и 'Last Review' заполним средними значениями
rev['TD Reviews'] = rev['TD Reviews'].fillna(round(rev['TD Reviews'].mean()))
rev['Last Review'] = rev['Last Review'].fillna(round(rev['Last Review'].mean()))

# Корректирую значения в столбце Review и переименовываю столбец, добавляю столбец 'Last Review'
df['Reviews'] = rev['TD Reviews']
df.rename(columns = {'Reviews': 'TD Reviews'}, inplace = True)
df['Last Review'] = rev['Last Review']
df.head(3)
response = requests.get('https://www.worldometers.info/world-population/population-by-country/')
response.status_code
# Данные о населении страны нашел на worldometers.info
url = 'https://www.worldometers.info/world-population/population-by-country/'
response = requests.get(url)
page = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
# Загружаем таблицу
all_blocks = page.find_all('div', class_='table-responsive')
data_html = all_blocks[0].find('table')
population_data = pd.read_html(str(data_html))[0]  
population_data.head(3)
# Создаю функцию, извлекающую данные из таблицы по названию города
def Population_Data(city):
    df = {}
    # В таблице приведены данные для страны, в исследуемом ДФ приведены города
    # Название страны получим используюя ранее созданный словарь cities_data
    country = cities_data[city][0]
    data = population_data[population_data['Country (or dependency)'] == country]
    df[country] = [data['Population (2020)'].values[0], # Население
                   data['Density (P/Km²)'].values[0],  # Плотность населения
                   float(data['Med. Age'].values[0]), # Медианный возраст населения
                   float(data['Urban Pop %'].values[0][:-2])] # Урбанизация
    return df

population_df = {}
for city in cities_data:
    if cities_data[city][0] not in population_df:
        population_df.update(Population_Data(city))
population_df
# Добавляем новые параметры в ДФ
df['Population'] = df['City'].apply(lambda x: population_df[cities_data[x][0]][0])
df['Density'] = df['City'].apply(lambda x: population_df[cities_data[x][0]][1])
df['Med. Age'] = df['City'].apply(lambda x: population_df[cities_data[x][0]][2])
df['Urban Pop. %'] = df['City'].apply(lambda x: population_df[cities_data[x][0]][3])
df.head(3)
# Дополняю ДФ данными о населении города из словаря cities_data
df['City Pop.'] = df['City'].apply(lambda x: cities_data[x][3])
df.head(3)
# Находим данные о размере ВВП стран по паритету покупательной способности
url_gdp = 'https://nonews.co/directory/lists/countries/gdp-ppp'
page = BeautifulSoup(requests.get(url_gdp, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
# Извлекаем таблицу со страницы сайта
all_blocks = page.find_all('div', class_='tab_widget wp_shortcodes_tabs')
data_html = all_blocks[0].find('table')
gdp_data = pd.read_html(str(data_html))[0]
gdp_data.head(3)
# Создаю функцию для извлечения данных из таблицы по наименованию города
def GDP_Data(city):
    country = cities_data[city][1]
    data = gdp_data[gdp_data['Страна'] == country]
    gdp = data['ВВП (дол. США)'].values[0]
    return int(''.join([x for x in (gdp).split()]))
# Добавляю параметр ВВП ('GDP') и ВВП на душу населения рассчитываю ('GDP/Pop')
df['GDP'] = df['City'].apply(lambda x: GDP_Data(x))
df['GDP/Pop.'] = df['GDP'] / df['Population']
df.head(3)
# Добавляю данные о количестве туристов 'Tourism' из словаря cities_data и туристов на душу населения 'Tourism/Pop.'
df['Tourism'] = df['City'].apply(lambda x: cities_data[x][4])
df['Tourism/Pop.'] = df['Tourism'] / df['Population']
df.head(3)
# Добавляем признак количество людей на ресторан в городе
df['People Per Rest.'] = df['City Pop.'] / df['Rest. Count']

# Добавляем признак количество туристов на ресторан в городе
df['Tourist Per Rest.'] = df['Tourism'] / df['Rest. Count']

# Добавляем признак количество отзывов на население города
df['Reviews per C.Pop.'] = df['Number of Reviews'] / df['City Pop.']

# Добавляем признак количество отзывов на количество туристов
df['Reviews per Tourism'] = df['Number of Reviews'] / df['Tourism']

df.head(3)
# Проверяем перед добавлением dummy-параметров
df.info()
# Проверим корреляцию признаков
C = df.corr(method='pearson')
plt.figure(figsize = (20, 20)) # размер графика
sns.heatmap(data = C, annot = True)
# Видно, что 'GDP' очень сильно скоррелирован с 'Population'
# Параметры 'Rest. Count', 'City Pop.' и 'Ranking' также сильно скоррелилованы между собой
# Применим метод PCA для создания нового признака из 'GDP' и 'Population'
from sklearn.decomposition import PCA
A = pd.concat([df['GDP'], df['Population']], axis = 1)
pca = PCA(1)
pca.fit(A)
pca_1 = pca.transform(A)
pca_1
# Применим метод PCA для создания нового признака из 'Rest. Count', 'City Pop.' и 'Ranking'
from sklearn.decomposition import PCA
A = pd.concat([df['Rest. Count'], df['City Pop.'], df['Ranking']], axis = 1)
pca = PCA(1)
pca.fit(A)
pca_2 = pca.transform(A)
pca_2
# Добавляем полученные параметры к ДФ
df_pca = pd.DataFrame({'PCA_1': list(pca_1), 'PCA_2': list(pca_2)})
df_pca['PCA_1'] = df_pca['PCA_1'].apply(lambda x: x[0])
df_pca['PCA_2'] = df_pca['PCA_2'].apply(lambda x: x[0])
df_pca.head(3)
# Добавляем к ДФ
df = pd.concat([df, df_pca], axis = 1)
df.head(3)
# Добавляем dummy-параметры городов
df_city = pd.get_dummies(df['City'])
df_city.head()
# Создадим dummy-параметры для кухонь
# cuisine_set - множество типов кухонь, созданное в разделе 6.3
# Создаем функцию для определения наличия кухни в перечне
def Cuisine_dummy(data, cuisine):
    if cuisine in data:
        return 1
    else:
        return 0

# Создаем пустой ДФ, куда сохраним dummy-параметры о кухнях
df_cuisines = pd.DataFrame()

# Добавляем dummy-параметры
for cuisine in cuisine_set:
#    print(cuisine)
#    print(df['Cuisine Style'][0])
#    print(Cuisine_dummy(df['Cuisine Style'][0], cuisine))
    df_cuisines[cuisine] = df['Cuisine Style'].apply(lambda x: Cuisine_dummy(x, cuisine))

df_cuisines.head(3)
# Объединяем в один ДФ
df_ml = pd.concat([df, df_city, df_cuisines], axis = 1)
df_ml.head(3)
df_ml.columns
df_ml.info()
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
X = df_ml[df_ml['Main'] == 1].drop(['Restaurant_id', 'City', 'Cuisine Style', 'Rating', 'GDP', 'Population',
             'Rest. Count', 'City Pop.', 'Ranking', 'Main', 'URL_TA'], axis = 1)
y = df_ml[df_ml['Main'] == 1]['Rating']

# Загружаем специальный инструмент для разбивки:
from sklearn.model_selection import train_test_split

# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 25% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = random_seed)
# Создаём модель
regr = RandomForestRegressor(n_estimators=100, verbose = 1, n_jobs = -1, random_state = random_seed)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)

# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
MAE = metrics.mean_absolute_error(y_test, y_pred)
print('MAE:', MAE)
print(y_pred, df['Rating'].unique(), sep = '\n')
# В столце 'Rating' приведены значения с точностю 0.5, таким образом для повышения точности
# прогноза приозведем округление результатов модели
# Создадим функцию округления с точность 0.5
def rating_pred(y_pred):
    y = y_pred // 0.25
    if y % 2 == 0:
        return y * 0.25
    else:
        return (y + 1) * 0.25


# Произведем округление
for i in range(len(y_pred)):
    y_pred[i] = rating_pred(y_pred[i])
# Проверим точность определения рейтинга с учетом округления
MAE_round = metrics.mean_absolute_error(y_test, y_pred)
print(f'Достигнуто значение MAE: {MAE_round}')
# Посмотрим на параметры, оказывающие наибольшее влияние на целевую переменную 
plt.figure(figsize = (10,10))
feat_importances = pd.Series(regr.feature_importances_, index = X.columns)
feat_importances.nlargest(20).plot(kind = 'barh')
# Создадим датафрейм с данными для передачи в модель для предсказания рейтингов
df_kt = df_ml[df_ml['Main'] == 0].drop(['Restaurant_id', 'City', 'Cuisine Style', 'Rating', 'GDP', 'Population',
             'Rest. Count', 'City Pop.', 'Ranking', 'Main', 'URL_TA'], axis = 1)
df_kt_y_pred = regr.predict(df_kt)
# Произведем округление
for i in range(len(df_kt_y_pred)):
    df_kt_y_pred[i] = rating_pred(df_kt_y_pred[i])
# Создадим датасет конечного результата submission_df
submission_df = pd.DataFrame()
# Запишем в него требуемые данные
submission_df['Restaurant_id'] = kt_data['Restaurant_id']
submission_df['Rating'] = df_kt_y_pred
submission_df
# Сохраним результат в файл
submission_df.to_csv('submission.csv', index = False)