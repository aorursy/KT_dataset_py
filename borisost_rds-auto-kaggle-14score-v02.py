import pandas as pd
import numpy as np
import requests
import csv
from random import randint
from time import sleep
import datetime
import re
from random import randint
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from lxml import html

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.max_columns', 100)

def autoru_page_parsing(url):
    # Функция собирает названия и ссылки объявлений на странице сайта Auto.ru 
    r = requests.get(url)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text)
        cars_list = soup.find_all('a', class_='Link ListingItemTitle-module__link')
        links = [car.get('href') for car in cars_list]
        
        three = html.fromstring(r.content)
        titles = three.xpath(
            '//a[@class="Link ListingItemTitle-module__link"]//text()'
        )
    else:
        print(f'Объявление по {url}, статус: {r.status_code}')
    
    return titles, links
        
        
def autoru_car_parsing(url):
    # Функция собирает данные по конкретным объявлениям
    # Сессия
    s = requests.Session()
    # Заголовки запроса
    s.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
        AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',\
        'Accept-Language': 'ru',
    }
    
    # Нулевая попытка - всего дается 10 (?)
    attempt = 1
    
    while attempt < 11:
        try:
            r = s.get(url)
            soup = BeautifulSoup(r.text)
            three = html.fromstring(r.content)
            
            # Цена автомобиля
            car_price = int(
                three.xpath(
                    '//span [@ class="OfferPriceCaption PriceUsedOffer-module__caption"]//text()'
                )[0].replace('\xa0', '')[:-1]
            )
            
            # Техничекие характеристики из базы данных по моделям
            model_techn_charact = soup.find(
                'a', class_='Link SpoilerLink CardCatalogLink SpoilerLink_type_default'
            ).get('href')
            
            # Основные и дополнительные характеристики автомобиля
            car_param_list = three.xpath('//ul [@ class="CardInfo"]//text()')

            # Описание автомобиля
            description = three.xpath('//div [@ class="CardDescription__textInner"]//text()')
            
            # Дополнительных параметров что-то не обнаружил
            adv_car_param_list = np.NaN
            
            # Список с описанием комплектации АМ
            complectation_list = three.xpath(
                '//div [@ class="CardComplectation__groups"]//text()'
            )
            
            break
        except:
            sleep(randint(1, 5))
            
        attempt += 1
    else:
        print(f'Попытки кончились: {url}')
        car_param_list = np.NaN
        adv_car_param_list = np.NaN
        description = np.NaN
        model_techn_charact = np.NaN
        complectation_list = np.NaN
        car_price = np.NaN
    
    if attempt != 1:
        print(f'Число попыток: {attempt}\t{url}')
    
    return car_param_list, adv_car_param_list, description, model_techn_charact, \
            complectation_list, car_price
# Сбор названий и ссылок на объявления на страницах
page_limit = 100
global_titles_list = []
global_links_list = []

for page in range(1, page_limit):
    url = f'https://auto.ru/rossiya/cars/bmw/used/?page={page}'
    titles_list_tmp, links_list_tmp = autoru_page_parsing(url)

    global_titles_list += titles_list_tmp
    global_links_list += links_list_tmp

    # Чтоб видеть процесс
    print(f'{100*(page+1)/page_limit:.2f}%', '*' * int(75 * (page+1)/page_limit), end='')
    print('\r', end='')
    
    sleep(randint(1, 3))

print(f'\nОбщее количество автомобилей: {len(global_titles_list)}')
# Запись списка ссылок со страниц в файл
with open(r'.\data\step01_autoru_main_links.csv', mode='w', encoding='utf8') as file:
    file_writer = csv.writer(file, delimiter=';')
    file_writer.writerow(['Model', 'Car_link'])
    for index in range(len(global_titles_list)):
        if global_titles_list[index] != None:
            file_writer.writerow([global_titles_list[index], 
                                  global_links_list[index]]) 

# Загрузка файла со ссылками
links_list = pd.read_csv(r'.\data\step01_autoru_main_links.csv', 
                         sep=';', encoding='utf8')
# Проверка на наличие дубликатов
links_list.duplicated().sum()
# Удаление дублей
links_list = links_list.drop_duplicates()
# Сбор данных по объявлениям. Запись данных в файл
with open(r'.\data\step02_autoru_cars.csv', mode='w', encoding='utf8') as file:
    file_writer = csv.writer(file, delimiter=';')
    file_writer.writerow(['Model',
                          'Car_link', 
                          'Model_parameters', 
                          'Adv_parameters', 
                          'Description', 
                          'Technical_characteristics_link', 
                          'Complectation', 
                          'Price'])
    for ii in range(links_list.shape[0]):
        model = links_list.iloc[ii].Model
        url = links_list.iloc[ii].Car_link
        file_writer.writerow([model, url] + list(autoru_car_parsing(url)))
        
        print(f'{100*(ii+1)/links_list.shape[0]:.2f}%', 
              '*' * int(100 * (ii+1)/links_list.shape[0]), end='')
        print('\r', end='')
        
#         if ii % 10 == 0:
#             print(f'Номер строки: {ii} - {ii/(links_list.shape[0] - 1)*100:.2f}%')
            
        sleep(randint(1, 3))

autoru_raw_df = pd.read_csv(r'.\data\step02_autoru_cars.csv', 
                            sep=';', 
                            encoding='utf8')
# Не нравится название столбца :)
autoru_raw_df = autoru_raw_df.rename(columns={'Model_parameters': 'Car_parameters'})
autoru_raw_df.head(5)
# Колонки
autoru_raw_df.columns
print(f'Количество дублей: {autoru_raw_df.duplicated().sum()}')
print(f'Количество NaN:\n {autoru_raw_df.isna().sum()}')
# Создание глубокой копии датасета
autoru_df_extend = autoru_raw_df.copy(deep=True)
# Удаление строк с неопределенными значениями в столбце + переименование строк
autoru_df_extend.dropna(subset=['Car_parameters'], axis=0, inplace=True)
autoru_df_extend.index = np.arange(0, autoru_df_extend.shape[0])

# Неопределенные параметры
print(f'Количество NaN:\n {autoru_df_extend.isna().sum()}')

def car_parameters_modification(param_str):
    # Преобразование и очистка основных параметров АМ. Словарь на выходе
    complicated_param = 'Бензин, газобаллонное оборудование'
    if complicated_param not in param_str:
        car_parameters_list = param_str[1:-1].\
                                        replace('\\xa0', '').\
                                        replace("'", '').\
                                        replace('/ ,', '/').\
                                        split(',')
    else:
        car_parameters_list = param_str[1:-1].\
                                        replace('Бензин, газобаллонное оборудование', 
                                                'Бензин + газобаллонное оборудование').\
                                        replace('\\xa0', '').\
                                        replace("'", '').\
                                        replace('/ ,', '/').\
                                        split(',')

    # Список названий парметров для конкретного АМ
    parameters_name_list = [
        car_parameters_list[i].strip() 
                         for i in range(len(car_parameters_list)) 
                             if i%2 == 0
    ]

    # Список значений параметров для конкретного АМ
    parameters_list = [
        car_parameters_list[i].strip() 
                         for i in range(len(car_parameters_list)) 
                             if i%2 == 1
    ]
    
    # Формирование словаря
    param_dict = dict(zip(parameters_name_list, parameters_list))
    
#     row.Car_parameters = [parameters_name_list, parameters_list]

    return param_dict
# Преобразование ячеек с параметрами АМ из строки в словарь
autoru_df_extend['Car_parameters'] = \
            autoru_df_extend['Car_parameters'].apply(car_parameters_modification)

# Вид произвольной ячейки Car_parameters
autoru_df_extend.loc[
    randint(0, autoru_df_extend.shape[0]), 
    'Car_parameters'].values()
# Формирование Series с множествами с названиями параметров
local_keys_sets = autoru_df_extend['Car_parameters'].apply(lambda cell: set(cell.keys()))
local_sets_list = list(local_keys_sets)
# Множество со всеми возможными параметрами АМ
glob_keys_set = local_sets_list[0]
for local_set in local_sets_list:
    glob_keys_set = glob_keys_set.union(local_set)
# Добавление столбцов с названиями параметров. Параметры по умолчанию - NaN
for name in glob_keys_set:
    autoru_df_extend[name] = np.nan
    
autoru_df_extend.head(5)
def car_param_filling(df):
    # Функция для заполнения параметров АМ - очень топорный подход, 
    # но ничего красивее пока не придумал
    
#     row = pd.DataFrame(row, columns=autoru_df.columns)
    for ii in range(df.shape[0]):
        param_list = df.loc[ii, 'Car_parameters'].keys()
    
        for name in param_list:
            df.loc[ii, name] = df.loc[ii, 'Car_parameters'][name]
        
    return df
# Заполнение столбцов
autoru_df_extend = car_param_filling(autoru_df_extend)

# Удаление столбца Car_parameters
autoru_df_extend.drop(['Car_parameters'], axis=1, inplace=True)

autoru_df_extend.head(5)
autoru_df_extend.groupby(['Обмен'])['Model'].count()
# Выбросы есть, их много - будут, возможно удалеть
fig = plt.figure(figsize=(12, 12))
sns.boxplot(x = autoru_df_extend['Кузов'], 
            y = autoru_df_extend['Price'], 
            hue = autoru_df_extend['Привод'])

autoru_df_extend['Двигатель'].unique()
autoru_df_extend['Двигатель'] = autoru_df_extend['Двигатель'].apply(lambda cell: cell.split('/'))
autoru_df_extend['engine_capacity'] = autoru_df_extend['Двигатель'].apply(lambda cell: cell[0].strip())
autoru_df_extend['enginePower'] = autoru_df_extend['Двигатель'].apply(lambda cell: cell[1].strip())
autoru_df_extend['fuelType'] = autoru_df_extend['Двигатель'].apply(lambda cell: cell[2].strip())
# Для электродвигателей требуется дополнительная обработка - у них нет Объема (= 0 л)! 
# Мощность двигателя в Ваттах не нужна - при необходимости можно получить из лошадей. 
# Номер строки с электродвигателем
id_elektro = autoru_df_extend[autoru_df_extend['fuelType'] == 'Электро'].index
id_elektro
# Преобразование строки с Электродвигателем
autoru_df_extend.loc[id_elektro, 'enginePower'] = \
                autoru_df_extend.loc[id_elektro, 'engine_capacity']
autoru_df_extend.loc[id_elektro, 'engine_capacity'] = '0 л'
# Проверка
autoru_df_extend.loc[id_elektro]
# Проверка Объема
autoru_df_extend['engine_capacity'].unique()
autoru_df_extend['engine_capacity'] = \
            autoru_df_extend['engine_capacity'].apply(lambda cell: float(cell.replace(' л', '')))
# Все возможные типы топлива
fuelType_uniq = autoru_df_extend['fuelType'].unique()
fuelType_uniq
# Формирование dummy- столбцов для топлива
autoru_df_extend = pd.concat([autoru_df_extend, 
                              pd.get_dummies(autoru_df_extend['fuelType'])], 
                             axis=1)
# Коэффициенты корреляции
fuelType_data_corr = autoru_df_extend[fuelType_uniq].corr()
fuelType_data_corr
# Тепловая карта
sns.set(font_scale=1)
plt.subplots(figsize=(10, 10))
sns.heatmap(fuelType_data_corr, square=True, 
           annot=True, fmt=".1f", linewidths=0.1, cmap="RdBu")
# Удаление столбцов fuelType
autoru_df_extend.drop(['fuelType'], axis=1, inplace=True) 
autoru_df_extend.head(5)
# Проверка столбца Мощность
autoru_df_extend['enginePower'].unique()
autoru_df_extend['enginePower'] = \
            autoru_df_extend['enginePower'].apply(lambda cell: int(cell.replace('л.с.', '')))
autoru_df_extend.head(5)
# Удаление столбца Двигатель
autoru_df_extend.drop(['Двигатель'], axis=1, inplace=True)
# Формирование столбца brand - чтоб не забыть, что речь о bmw
autoru_df_extend['brand'] = 'BMW'
# Структура наименования АМ, например: BMW 7 серия Long VI (G11/G12) 730Ld xDrive
# * BMW - марка
# * 7 серия Long VI - модель
# * G11/G12 - кузов
# * 730Ld xDrive - двигатель
uniq_list = autoru_df_extend['Model'].unique()
uniq_list
def engine_search(cell):
    # Поиск основных типов двигателей
    engine_pattern = re.compile('\sM?\d+[.]?\d{1,3}[ALa-z]*\s?[xs]?D?r?i?v?e?')
    
    engine_model = re.findall(engine_pattern, cell)
    
    if len(engine_model) != 0:
        return engine_model[0].strip()
    else:
        return np.nan
# Поиск основных типов двигателей
autoru_df_extend['engine'] = autoru_df_extend['Model'].apply(engine_search)
# Уникальные определенные двигатели (всего 152шт)
autoru_df_extend['engine'].unique()
# Количество моделей с неопределенными двигателями
na_eng = autoru_df_extend['engine'].isna()
na_eng.sum()
# Уникальные модели с неопределенными двигателями (всего 37)
na_engine_df['Model'].unique()
# Количество каждого типа уникальных моделей - всего 37 моделей, 
# максимальное количество автомобилей с неопределенным двигателем- 33
na_engine_df.groupby(['Model'])['Model'].count().sort_values()
# Число определенных двигателей, установленных на АМ менее 50 раз
# - в текущей постановке - редких определенных двигателей
sum(autoru_df_extend.groupby(['engine'])['engine'].count() < 50)
# Сортировка по количеству машин с определенным типом двигатея - наиболее часто встречающиеся
autoru_df_extend.groupby(['engine'])['engine'].count().sort_values(ascending=False).head(30)
# Series с логическими переменными - марки двигателей (число машин > 50)
over50_engines = autoru_df_extend.groupby(['engine'])['engine'].count() > 50

# Число моделей с определенным типом двигателя, количество машин > 50 
# - столько будет добавлено столбцов
sum(over50_engines)
over50_engines
# Список двигателей для dummy
over50_engines_list = [over50_engines.index[i] for i in range(over50_engines.shape[0]) 
                                               if over50_engines[i]]
over50_engines_list
def less50_engines(cell):
    # Преобразует ячейку в nan если число машин с этим двигателем <50
    if cell not in over50_engines_list:
        return np.nan
    else:
        return cell
# Преобразование ячейки в nan, если число машин с этим двигателем <50
autoru_df_extend['engine'] = autoru_df_extend['engine'].apply(less50_engines)

# Замена NaN в столбце engine на otherEngines
autoru_df_extend['engine'].fillna('otherEngines', inplace=True)

# Формирование dummy
autoru_df_extend = \
    pd.concat([autoru_df_extend, pd.get_dummies(autoru_df_extend['engine'])], axis=1)
# Проверка
autoru_df_extend
# Удаляю столбец engine
autoru_df_extend.drop(['engine'], axis=1, inplace=True)

# Все значения
autoru_df_extend['Владельцы'].unique()
# Добавление пробела
autoru_df_extend['Владельцы'] = \
        autoru_df_extend['Владельцы'].apply(lambda cell: cell.replace('влад', ' влад'))

# Формирование dummy- столбцов для владельцев
autoru_df_extend = pd.concat([autoru_df_extend, pd.get_dummies(autoru_df_extend['Владельцы'])], axis=1)

# Удаление столбца Владельцы
autoru_df_extend.drop(['Владельцы'], axis=1, inplace=True)
# Количество автомобилей по числу владелеьцев
autoru_df_extend[['1 владелец', '2 владельца', '3 или более']].sum()
autoru_df_extend.head(5)

# Все значения
unique_list = autoru_df_extend['Владение'].unique()
unique_list
def ownership_month(cell):
    # Функция преобразует сроку с временем владения в целое число месяцев
    
    # Шаблон для поиска времени владения (год и месяц)
    ownership_pattern_year = re.compile('\d+\s[л|г]')
    ownership_pattern_month = re.compile('\d+\s[м]')
    
    # Проверка, что cell - строка
    if isinstance(cell, str):
        # Поиск и вычисление года
        try:
            year = int(
                re.findall(
                    ownership_pattern_year, 
                    cell
                )[0][:-2]
            )
        except:
            year = 0
            
        # Поиск месяцев
        try:
            month = int(
                re.findall(
                    ownership_pattern_month, 
                    cell
                )[0][:-2]
            )
        except:
            month = 0
        
        return int(12 * year + month)
    else:
        return np.nan
# Преобразование столбца
autoru_df_extend['Владение'] = autoru_df_extend['Владение'].apply(ownership_month)

# Проверка
autoru_df_extend['Владение']
autoru_df_extend.head(5)
# Пропущенные значения
autoru_df_extend['Владение'].isna().sum()
# Вообще, пропуски надо бы обработать. Но проще будет удалить столбец. Все что выше - просто удалить жалко
autoru_df_extend_drop.drop(['Владение'], axis=1, inplace=True)

# Преобразование формата в int
autoru_df_extend['год выпуска'] = autoru_df_extend['год выпуска'].astype(int)

# Уникальные даты
autoru_df_extend['год выпуска'].unique()
def car_age_weeks(cell):
    # Преобразование года выпуска АМ в возраст в неделях
    
    # Если АМ произведен в 2020 - производство - начало года
    if cell == 2020:
        prod_date = datetime.date(2020, 1, 1)
    else:
        # Дата производства
        prod_date = datetime.date(cell, 12, 31)
    
    # Сегодняшняя дата
    date_now = datetime.datetime.now().date()
    
    # Разница в днях
    delta_date = date_now - prod_date
    
    # Количество полных недель
    return int(delta_date.days / 7)
autoru_df_extend['carAge'] = autoru_df_extend['год выпуска'].apply(car_age_weeks)

# Удаление столбца Год выпуска
autoru_df_extend.drop(['год выпуска'], axis=1, inplace=True)

autoru_df_extendюруфв(5)

# Уникальные типы кузова
autoru_df_extend['Кузов'].unique()
# Количество автомобилей с кузовами определенного типа 
# Для обработки берем типы кузовов > 50 АМ
autoru_df_extend.groupby(['Кузов'])['Кузов'].count().sort_values(ascending=False)
# Список кузовов с АМ > 50
over50_body = autoru_df_extend.groupby(['Кузов'])['Кузов'].count() > 50

# Список кузовов
over50_body_list = [over50_body.index[i] for i in range(over50_body.shape[0])
                                           if over50_body[i]]
over50_body_list
def body_type_column(cell):
    # Формирует столбец bodyType. Если менее 50 АМ в таком кузове - otherBodyType
    if cell not in over50_body_list:
        return np.nan
    else:
        return cell
# Формирование столбца bodyType
autoru_df_extend['bodyType'] = autoru_df_extend['Кузов'].apply(body_type_column)

autoru_df_extend['bodyType'].unique()
# Замена NaN в столбце bodyType на otherBodyType
autoru_df_extend['bodyType'].fillna('otherBodyType', inplace=True)
# Формирование dummy
autoru_df_extend = \
    pd.concat([autoru_df_extend, pd.get_dummies(autoru_df_extend['bodyType'])], axis=1)
# Удаление столбцов Кузов и bodyType
autoru_df_extend.drop(['Кузов', 'bodyType'], axis=1, inplace=True)
autoru_df_extend.head(5)

# Наверное можно было бы регион проверить... Удаляем - в тестовой нет
autoru_df_extend['Госномер']
# Наличие VIN - это хорошо. Удаляем - в тестовой нет
autoru_df_extend['VIN']
# Интересно, конечно, влияет ли на что-нибудь VIN - времени нет с этим копаться...
autoru_df_extend.groupby(['VIN']).VIN.count()
# Столбец вообще пустой. Удаляем - в тестовой нет
autoru_df_extend['Кузов №'].isna().sum()
# Удаляю все три столбца
autoru_df_extend.drop(['Кузов №', 'VIN', 'Госномер'], axis=1, inplace=True)
autoru_df_extend.head(5)

# Все автомобили растоможены, пропусков нет
autoru_df_extend['Таможня'].unique()
# Все автомобили не требуют ремонта
autoru_df_extend['Состояние'].unique()
# Удаление
autoru_df_extend.drop(['Таможня', 'Состояние'], axis=1, inplace=True)

autoru_df_extend['Руль'].unique()
# Целая ОДНА машина с правым рулем. Удалю позже
autoru_df_extend.groupby(['Руль'])['Руль'].count()
# Столбец переименовать в Левый Руль. Если да-1, если нет - 0
autoru_df_extend = autoru_df_extend.rename(columns={'Руль': 'Левый руль'})

# Замена значений Левый-1, Правый-0
autoru_df_extend.loc[autoru_df_extend['Левый руль'] == 'Левый', 'Левый руль'] = 1
autoru_df_extend.loc[autoru_df_extend['Левый руль'] == 'Правый', 'Левый руль'] = 0

autoru_df_extend['Левый руль'].unique()

autoru_df_extend.groupby(['Цвет'])['Цвет'].count().sort_values(ascending=False)
# Список цветов с АМ > 50
over50_color = autoru_df_extend.groupby(['Цвет'])['Цвет'].count() > 50

# Список цветов
over50_color_list = [over50_color.index[i] for i in range(over50_color.shape[0])
                                           if over50_color[i]]
over50_color_list
def color_column(cell):
    # Формирует столбец color. Меньше 50 - Nan
    if cell not in over50_color_list:
        return np.nan
    else:
        return cell
# Формирование столбца color
autoru_df_extend['color'] = autoru_df_extend['Цвет'].apply(color_column)

# Все что NaN - otherColor
autoru_df_extend['color'].fillna('otherColor', inplace=True)

# Формирование dummy
autoru_df_extend = \
    pd.concat([autoru_df_extend, pd.get_dummies(autoru_df_extend['color'])], axis=1)

# Удаление столбцов Цвет и color
autoru_df_extend.drop(['Цвет', 'color'], axis=1, inplace=True)
autoru_df_extend.head(5)

autoru_df_extend['ПТС'].unique()
autoru_df_extend.groupby(['ПТС'])['ПТС'].count()
# Столбец переименуется: Оригинал ПТС. 
autoru_df_extend = autoru_df_extend.rename(columns={'ПТС': 'Оригинал ПТС'})

# Если Оригинал=1, Дубликат=0
autoru_df_extend.loc[autoru_df_extend['Оригинал ПТС']=='Оригинал', 'Оригинал ПТС'] = 1
autoru_df_extend.loc[autoru_df_extend['Оригинал ПТС']=='Дубликат', 'Оригинал ПТС'] = 0

autoru_df_extend.head(3)

autoru_df_extend['Коробка'].unique()
autoru_df_extend.groupby(['Коробка'])['Коробка'].count()
# Пропусков нет
autoru_df_extend['Коробка'].isna().sum()
# Dummy переменные
autoru_df_extend = \
    pd.concat([autoru_df_extend, pd.get_dummies(autoru_df_extend['Коробка'])], axis=1)

# Удаление столбца Коробка
autoru_df_extend.drop(['Коробка'], axis=1, inplace=True)

autoru_df_extend.head(3)

autoru_df_extend['Привод'].unique()
autoru_df_extend.groupby(['Привод'])['Привод'].count()
# Пропусков нет
autoru_df_extend['Привод'].isna().sum()
# Формирование dummy
autoru_df_extend = \
    pd.concat([autoru_df_extend, pd.get_dummies(autoru_df_extend['Привод'])], axis=1)

# Удаление столбца Привод
autoru_df_extend.drop(['Привод'], axis=1, inplace=True)

autoru_df_extend = autoru_df_extend.rename(columns={'Пробег': 'mileage'})

autoru_df_extend['mileage']
# Оставляем только цифры
autoru_df_extend['mileage'] = autoru_df_extend['mileage'].apply(lambda cell: int(cell[:-2]))
autoru_df_extend.head(5)

autoru_df_extend.info()
# Категориальные признаки
drop_columns = list(autoru_df_extend.select_dtypes('object').columns)
drop_columns
# Копия DF
autoru_df_extend_drop = autoru_df_extend.copy(deep=True)
# Удаление категориальных и пустых признаков (Adv_parameters). 
# К сожалению столбцы с комплектацией и описанием остались необработанными
autoru_df_extend_drop.drop(drop_columns + ['Adv_parameters'], axis=1, inplace=True)
autoru_df_extend_drop.info()

with open(r'.\data\autoru_df_f_ml_20200719.csv', mode='w', encoding='utf8') as file:
    file_writer = csv.writer(file, delimiter=';')
    file_writer.writerow(autoru_df_extend_drop.columns)
    for ii in range(autoru_df_extend_drop.shape[0]):
        file_writer.writerow(autoru_df_extend_drop.loc[ii])

data_frame_raw = pd.read_csv('.\\data_autoru_ML\\autoru_df_f_ml_20200719.csv', sep=';', encoding='utf8')
data_frame_raw
# Колонки датафрейма
df_columns = data_frame_raw.columns
df_columns
# Глубокая копия датасета
data_frame= data_frame_raw.copy(deep=True)

data_frame['Другое_топливо'] = data_frame['Бензин + газобаллонное оборудование'] + \
                                data_frame['Гибрид'] + data_frame['Электро']
# Удаление 'Бензин + газобаллонное оборудование', 'Гибрид', 'Электро'
data_frame.drop(['Бензин + газобаллонное оборудование', 'Гибрид', 'Электро'], 
                axis=1, inplace=True)
# Изменение порядка столбцов
data_frame = data_frame[['Price', 'Оригинал ПТС', 'Левый руль', 'mileage', 'engine_capacity',
       'enginePower', 'Бензин', 'Дизель', 'Другое_топливо', '116i', '20d', '20d xDrive',
       '20i xDrive', '25d', '30d', '318i', '320d xDrive', '320i',
       '320i xDrive', '35i', '40d', '520d', '520d xDrive', '520i',
       '528i xDrive', '530d xDrive', '730Ld xDrive', 'M50d', 'otherEngines',
       '1 владелец', '2 владельца', '3 или более', 'carAge', 'внедорожник 5 дв.', 
       'купе', 'лифтбек', 'седан', 'хэтчбек 5 дв.', 'otherBodyType',
       'белый', 'коричневый', 'красный', 'серебристый', 'серый',
       'синий', 'чёрный', 'otherColor', 'автоматическая', 'механическая', 'роботизированная',
       'задний', 'передний', 'полный']]

data_frame

# Добавление признака - средний пробег: пробег/возраст [км/неделя]. Его логарифм
data_frame['log_mileage_average'] = np.log(data_frame['mileage'] / (data_frame['carAge'] + 1))
data_frame['log_mileage_average']

df_columns = data_frame.columns
df_columns
poly_features_list = ['mileage', 'enginePower', 'carAge']

# Полиномиальные признаки. Результаты представляются в виде массива со столбцами: [a, b, a^2, ab, b^2]
pf = PolynomialFeatures(2, include_bias=False)
poly_features_df = pd.DataFrame(
    pf.fit_transform(data_frame[poly_features_list]), 
    columns=pf.get_feature_names(poly_features_list)
)

# Удаление существующих столбцов
poly_features_df.drop(poly_features_list, axis=1, inplace=True)

# Объединение 
data_frame = pd.concat([data_frame, poly_features_df], axis=1)

data_frame
sns.set()
sns.distplot(data_frame.Price)
# Гиперболический признак - возраст автомобиля
data_frame['1/carAge'] = 1 / data_frame['carAge']

# У целевой переменной Price большие перепады - можно попробовать прологорифмировать
data_frame.Price = data_frame.Price.apply(lambda cell: np.log(cell + 1))
sns.set()
sns.distplot(data_frame.Price)

sns_plot = sns.pairplot(data_frame, vars=['Price', 'mileage', 'engine_capacity',
       'enginePower', 'carAge', 'log_mileage_average', 'mileage^2', 'mileage enginePower', 'mileage carAge', 'enginePower^2', 'enginePower carAge', 'carAge^2', '1/carAge'])

def outliers_iqr(ys):
    # Определяет номера значений с отклонением больше чем iqr
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - 1.5*iqr
    upper_bound = quartile_3 + 1.5*iqr
    return np.where((ys > upper_bound) | (ys < lower_bound))[0]
fig = plt.figure(figsize=(7, 7))
plt.grid(True)
plt.title('Price')
sns.boxplot(x = data_frame['Price'])
outliers_data_price = outliers_iqr(data_frame['Price'])
outliers_data_price
# Выбросы пробега
fig = plt.figure(figsize=(7, 7))
plt.grid(True)
sns.boxplot(x = data_frame['mileage'])
outliers_data_mileage = outliers_iqr(data_frame['mileage'])
outliers_data_mileage
# Строки с пробегом в 1км - неправдоподобно
low_mileage = data_frame[data_frame['mileage'] == 1].index
low_mileage
# Выбросы среднего пробега
fig = plt.figure(figsize=(7, 7))
plt.grid(True)
sns.boxplot(x = data_frame['log_mileage_average'])
outliers_data_mileage_average = outliers_iqr(data_frame['log_mileage_average'])
outliers_data_mileage_average
# Выбросы квадрата возраста
fig = plt.figure(figsize=(7, 7))
plt.grid(True)
sns.boxplot(x = data_frame['carAge^2'])
outliers_carAge_sq = outliers_iqr(data_frame['carAge^2'])
outliers_carAge_sq
# Объединенный список выбросов
outliers_data_sum = set(np.hstack((outliers_data_price, 
                                   outliers_data_mileage, 
                                   low_mileage, 
                                   outliers_data_mileage_average, 
                                   outliers_carAge_sq)))
len(outliers_data_sum)
# Удаление - не реализовано 
# data_frame.drop(outliers_data_sum, axis=0, inplace=True)
# Графики без выбросов
sns_plot = sns.pairplot(data_frame, vars=['Price', 'mileage', 'engine_capacity',
       'enginePower', 'carAge', 'log_mileage_average', 'mileage^2', 'mileage enginePower', 'mileage carAge', 'enginePower^2', 'enginePower carAge', 'carAge^2', '1/carAge'])

# Колонки датафрейма
df_columns = data_frame.columns
df_columns
# Коррекция индексов после удаления выбросов
data_frame.index = np.arange(0, data_frame.shape[0])
# Разбиение
X = data_frame[df_columns[1:]]
Y = data_frame['Price']
x_corr = X.corr()
x_corr
sns.set(font_scale=5)
plt.subplots(figsize=(150, 150))
sns.heatmap(x_corr, square=True, 
           annot=True, fmt=".1f", linewidths=0.1, cmap="RdBu")
# Названия столбцов для удаления из DF
drop_list = [
    'carAge', 
    'engine_capacity', 
    '1 владелец', 
    'Бензин', 
    'полный', 
    'автоматическая', 
    'внедорожник 5 дв.', 
    '116i', 
    'Левый руль',
]
# Удаление столбцов с большими коэффициентами корреляции
X = X.drop(drop_list, axis=1)

# Разбиение выборки 
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size= 0.2)
X_train

def minmax_normalization(df):
    # Функция проводит minmax нормализацию DF
    sc = MinMaxScaler()
    df_norm = sc.fit_transform(df)

    return df_norm

def mean_absolute_percent_error(y_true, y_pred):
    # Расчет значения метрики 
    
    return 100 * np.mean( np.abs(y_true - y_pred) / y_true )

X_train = np.hstack([np.ones(X_train.shape[0])[:, np.newaxis], X_train])
X_valid = np.hstack([np.ones(X_valid.shape[0])[:, np.newaxis], X_valid])

# X_train_std = np.hstack([np.ones(X_train_std.shape[0])[:, np.newaxis], X_train_std])
# X_valid_std = np.hstack([np.ones(X_valid_std.shape[0])[:, np.newaxis], X_valid_std])
# Линейная регрессия
myModel = LinearRegression()
myModel.fit(X_train, Y_train)
# Предсказание
y_pred = myModel.predict(X_valid)

# Расчет метрики. Предсказанные и реальные значения цены - под логарифмом!!
mean_absolute_percent_error(
    np.exp(Y_valid) - 1, 
    np.exp(y_pred) - 1
)

kf = KFold(n_splits=3, shuffle=True)

# Список значение метрик для разных наборов
kfold_errors_list = []
kfold_errors_norm_list = []

for train_index, valid_index in kf.split(X):
    
    # Разбиение выборок
    X_train_kfold, X_valid_kfold = X.loc[train_index], X.loc[valid_index]
    Y_train_kfold, Y_valid_kfold = Y.loc[train_index], Y.loc[valid_index]
    
    # Нормальзация
    X_train_kfold_norm = minmax_normalization(X_train_kfold)
    X_valid_kfold_norm = minmax_normalization(X_valid_kfold)
    
    # Добавление единичных столбцов
    X_train_kfold = np.hstack(
        [np.ones(X_train_kfold.shape[0])[:, np.newaxis], 
         X_train_kfold]
    )
    X_valid_kfold = np.hstack(
        [np.ones(X_valid_kfold.shape[0])[:, np.newaxis], 
         X_valid_kfold]
    )
    
    # Линейная регрессия 
    myModel_kfold = LinearRegression()
    myModel_kfold.fit(X_train_kfold, Y_train_kfold)
    
    # Предсказанное значение логарифма цены
    Y_pred_kfold = myModel_kfold.predict(X_valid_kfold)

    kfold_errors_list.append(
        mean_absolute_percent_error(
            np.exp(Y_valid_kfold) - 1, 
            np.exp(Y_pred_kfold) - 1)
    )
    
    # ==================================================
    
    # Добавление единичных столбцов - НОРМАЛИЗОВАННАЯ
    X_train_kfold_norm = np.hstack(
        [np.ones(X_train_kfold_norm.shape[0])[:, np.newaxis], 
         X_train_kfold_norm]
    )
    X_valid_kfold_norm = np.hstack(
        [np.ones(X_valid_kfold_norm.shape[0])[:, np.newaxis], 
         X_valid_kfold_norm]
    )
    
    # Линейная регрессия для НОРМАЛИЗОВАННОЙ выборки
    myModel_kfold_norm = LinearRegression()
    myModel_kfold_norm.fit(X_train_kfold_norm, Y_train_kfold)
    
    # Предсказанное значение логарифма цены
    Y_pred_kfold_norm = myModel_kfold.predict(X_valid_kfold_norm)

    kfold_errors_norm_list.append(
        mean_absolute_percent_error(
            np.exp(Y_valid_kfold) - 1, 
            np.exp(Y_pred_kfold_norm) - 1)
    )

#     kfold_errors_list.append(
#         mean_absolute_percent_error(
#             Y_valid_kfold, 
#             Y_pred_kfold))

    print(f'MAPE: {kfold_errors_list[-1]}\tMAPE_norm:{kfold_errors_norm_list[-1]}')

print('='*50)
print(f'Average\nMAPE: {kfold_errors_list[-1]}\tMAPE_norm:{kfold_errors_norm_list[-1]}')

kf = KFold(n_splits=3, shuffle=True)

# Список значение метрик для разных наборов
kfold_errors_list = []
kfold_errors_norm_list = []

for train_index, valid_index in kf.split(X):
    
    # Разбиение выборок
    X_train_kfold, X_valid_kfold = X.loc[train_index], X.loc[valid_index]
    Y_train_kfold, Y_valid_kfold = Y.loc[train_index], Y.loc[valid_index]
    
    RFRModel_kfold = RandomForestRegressor(n_estimators=100, max_depth=20)
    RFRModel_kfold.fit(X_train_kfold, Y_train_kfold)
    
    # Предсказанное значение логарифма цены
    Y_predict_kfold = RFRModel_kfold.predict(X_valid_kfold)

    kfold_errors_list.append(
        mean_absolute_percent_error(
            np.exp(Y_valid_kfold) - 1, 
            np.exp(Y_predict_kfold) - 1)
    )
    
    X_train_kfold_norm = minmax_normalization(X_train_kfold)
    X_valid_kfold_norm = minmax_normalization(X_valid_kfold)
    
    # Случайный лес 
    RFRModel_kfold_norm = RandomForestRegressor(n_estimators=50, max_depth=25)
    RFRModel_kfold_norm.fit(X_train_kfold_norm, Y_train_kfold)
    
    # Предсказанное значение логарифма цены
    Y_pred_kfold_norm = RFRModel_kfold_norm.predict(X_valid_kfold_norm)

    kfold_errors_norm_list.append(
        mean_absolute_percent_error(
            np.exp(Y_valid_kfold) - 1, 
            np.exp(Y_pred_kfold_norm) - 1)
    )

#     kfold_errors_list.append(
#         mean_absolute_percent_error(
#             Y_valid_kfold, 
#             Y_pred_kfold))
    
    print(f'MAPE: {kfold_errors_list[-1]}\tMAPE_norm:{kfold_errors_norm_list[-1]}')

print('='*50)
print(f'Average\nMAPE: {kfold_errors_list[-1]}\tMAPE_norm:{kfold_errors_norm_list[-1]}')

df_test = pd.read_csv(r'./data/test.csv')
df_test.head(5)
df_test.shape
# Удаление столбцов, которые далее не участвуют
df_test.drop(['description', 
              'brand', 
              'vehicleConfiguration', 
              'Комплектация', 
              'Владение', 
              'numberOfDoors', 
              'Таможня', 
              'Руль'], axis=1, inplace=True)
# Названия колонок
df_test.columns
# Пропуски в данных 
df_test.info()

# Уникальные названия
uniq_name_list = df_test['name'].unique()
uniq_name_list
# Убрать пробел перед xDrive, л.с. и т.д. чтобы потом разбить по пробелам
df_test['name'] = df_test['name'].apply(lambda cell: cell.replace(' xDrive', 'xDrive'))
df_test['name'] = df_test['name'].apply(lambda cell: cell.replace(' sDrive', 'sDrive'))
df_test['name'] = df_test['name'].apply(lambda cell: cell.replace(' л.', 'л.'))
df_test['name'] = df_test['name'].apply(lambda cell: cell.replace('Active Hybrid 7L', 
                                                                  'ActiveHybrid7L'))
df_test['name'] = df_test['name'].apply(lambda cell: cell.replace('Competition Package', 
                                                                  'CompetitionPackage'))
df_test['name'] = df_test['name'].apply(lambda cell: cell.replace(' кВт', 'кВт'))

# Удаление 4WD
df_test['name'] = df_test['name'].apply(lambda cell: cell.replace(' 4WD', ''))

# Разделение данных
df_test['name'] = df_test['name'].apply(lambda cell: cell.split())

df_test['engine'] = df_test['name'].apply(lambda cell: cell[0])
uniq_engine_list = df_test['engine'].unique()
uniq_engine_list
def engine_str_transform(cell):
    # Функция почему-то не работает. Должна была исправлять косяки 
    # df (перестовлять xDrive назад) + добавлять пробелы там гда надо
    # Но она не работает! replace без функции -норм
    
    eng_pattern = re.compile('[xs]?D?r?i?v?e?')
    eng_model = re.match(eng_pattern, cell)
    
#     print(len(eng_model[0]))
    
    if eng_model[0] == 'xDrive' or eng_model[0] == 'sDrive':
        cell_transform = cell.replace(eng_model[0], '') + f' {eng_model[0]}'
    else:
        cell_transform = cell.replace('xDrive', ' xDrive')
        cell_transform = cell.replace('sDrive', ' sDrive')
        cell_transform = cell.replace('CompetitionPackage', 'Competition Package')
        cell_transform = cell.replace('ActiveHybrid7L', 'Active Hybrid 7L')
#         print(eng_model[0] + 'xdrive')
    return cell_transform
df_test['engine'] = df_test['engine'].apply(engine_str_transform)
df_test['engine'].unique()
df_test['engine'] = df_test['engine'].apply(lambda cell: cell.replace('xDrive', ' xDrive'))
df_test['engine'] = df_test['engine'].apply(lambda cell: cell.replace('sDrive', ' sDrive'))
df_test['engine'] = df_test['engine'].apply(lambda cell: 
                                            cell.replace('CompetitionPackage', 
                                                         'Competition Package'))
df_test['engine'] = df_test['engine'].replace('ActiveHybrid', 'Active Hybrid')
# Убираются двойные пробелы
df_test['engine'] = df_test['engine'].apply(lambda cell: cell.replace('  ', ' '))
df_test['engine'].unique()
# Наиболее распространенные двигатели в тестовой выборке
df_test.groupby(['engine'])['engine'].count().sort_values(ascending=False).head(30)

df_test_col_name = df_test.columns
df_test_col_name
df_test_col_name_new = df_test_col_name + '_test'
df_test_col_name_new
rename_dict = {df_test_col_name[i]: df_test_col_name_new[i] 
               for i in range(len(df_test_col_name))}
rename_dict
# Переименование столбцов тестовой выборки
df_test = df_test.rename(columns=rename_dict)
df_test.head(5)
# Названия столбцов из обучающей выбоки. Столбец mileage_average - под логарифмом
train_df_columns = X.columns
train_df_columns
# Добавление столбцов из учебной выборки в тестовую
for i in range(len(train_df_columns)):
    df_test[train_df_columns[i]] = np.nan
    
df_test.head(5)
df_test.columns

bodyType_unique = df_test.bodyType_test.unique()
bodyType_unique
df_test.groupby(['bodyType_test'])['bodyType_test'].count().sort_values(ascending=False)
train_bodyType_list = ['купе', 'лифтбек', 'седан', 'хэтчбек 5 дв.', 'otherBodyType']
# Распределение типов кузовов
for i in range(df_test.shape[0]):
    body_name = df_test.loc[i, 'bodyType_test']
    if body_name in train_bodyType_list:
        df_test.loc[i, body_name] = 1
    elif body_name not in train_bodyType_list and body_name != 'внедорожник 5 дв.':
        df_test.loc[i, 'otherBodyType'] = 1
df_test[train_bodyType_list] = df_test[train_bodyType_list].fillna(0)
# Проверка
df_test[train_bodyType_list].sum()
df_test[train_bodyType_list].head(5)
# Удаление столбца bodyType_test
df_test.drop(['bodyType_test'], axis=1, inplace=True)
df_test.head(5)

train_color_list = ['белый', 'коричневый', 'красный', 
                       'серебристый', 'серый', 'синий', 'чёрный', 'otherColor']
df_test.groupby(['color_test'])['color_test'].count().sort_values(ascending=False)
# Распределение цветов
for i in range(df_test.shape[0]):
    color_name = df_test.loc[i, 'color_test']
    if color_name in train_color_list:
        df_test.loc[i, color_name] = 1
    elif color_name not in train_color_list:
        df_test.loc[i, 'otherColor'] = 1
df_test.head(5)
df_test[train_color_list] = df_test[train_color_list].fillna(0)
df_test[train_color_list]
# Удаление столбца color_test
df_test.drop(['color_test'], axis=1, inplace=True)

df_test.groupby(['fuelType_test'])['fuelType_test'].count().sort_values(ascending=False)
df_test.loc[df_test['fuelType_test'] == 'бензин', 'fuelType_test'] = 'Бензин'
df_test.loc[df_test['fuelType_test'] == 'дизель', 'fuelType_test'] = 'Дизель'
train_fuelType_list = ['Дизель', 'Другое_топливо']
# Распределение типов топлива
for i in range(df_test.shape[0]):
    fuel_name = df_test.loc[i, 'fuelType_test']
    if fuel_name in train_fuelType_list:
        df_test.loc[i, fuel_name] = 1
    elif fuel_name not in train_fuelType_list and fuel_name != 'Бензин':
        df_test.loc[i, 'Другое_топливо'] = 1
df_test.head(5)
# Удаление столбца fuelType_test
df_test.drop(['fuelType_test'], axis=1, inplace=True)
df_test[train_fuelType_list] = df_test[train_fuelType_list].fillna(0)

# Удаление столбца modelDate_test
df_test.drop(['modelDate_test'], axis=1, inplace=True)
df_test.head(5)

# Удаление столбца name_test
df_test.drop(['name_test'], axis=1, inplace=True)
df_test.head(5)

df_test['productionDate_test'].unique()
df_test['productionDate_test'] = df_test['productionDate_test'].astype(int)
def car_age_weeks(cell):
    # Преобразование года выпуска АМ в возраст в неделях
    
    # Если АМ произведен в 2020 - производство - начало года
    if cell == 2020:
        prod_date = datetime.date(2020, 1, 1)
    else:
        # Дата производства
        prod_date = datetime.date(cell, 12, 31)
    
    # Сегодняшняя дата
    date_now = datetime.datetime.now().date()
    
    # Разница в днях
    delta_date = date_now - prod_date
    
    # Количество полных недель
    return int(delta_date.days / 7)
# Столбец CarAge (потом будет удален) для вычисления среднего пробега
df_test['carAge'] = df_test['productionDate_test'].apply(car_age_weeks)
df_test['carAge']
# Удаление столбца productionDate_test
df_test.drop(['productionDate_test'], axis=1, inplace=True)
df_test.head(5)

df_test['mileage'] = df_test['mileage_test']
df_test['mileage'] = df_test['mileage'].astype(float)
df_test['mileage']
# Удаление столбца engineDisplacement_test
df_test.drop(['mileage_test'], axis=1, inplace=True)

df_test['log_mileage_average'] = np.log(df_test['mileage'] / (df_test['carAge'] + 1))
df_test.head(5)

df_test.groupby(['vehicleTransmission_test'])['vehicleTransmission_test'].count().sort_values(ascending=False)
train_transType_list = ['механическая', 'роботизированная']
# Распределение типов привода
for i in range(df_test.shape[0]):
    trans_name = df_test.loc[i, 'vehicleTransmission_test']
    if trans_name in train_transType_list:
        df_test.loc[i, trans_name] = 1
df_test.head(5)
# Удаление столбца vehicleTransmission_test
df_test.drop(['vehicleTransmission_test'], axis=1, inplace=True)
df_test[train_transType_list] = df_test[train_transType_list].fillna(0)

# Удаление столбца engineDisplacement_test
df_test.drop(['engineDisplacement_test'], axis=1, inplace=True)

df_test['enginePower_test'].unique()
df_test['enginePower_test'] = df_test['enginePower_test'].apply(
    lambda cell: int(cell.split()[0]))
df_test['enginePower'] = df_test['enginePower_test']
df_test.head(5)
df_test.drop(['enginePower_test'], axis=1, inplace=True)

df_test.groupby(['Привод_test'])['Привод_test'].count()
train_transmission_list = ['задний', 'передний']
# Распределение приводов
for i in range(df_test.shape[0]):
    trans_name = df_test.loc[i, 'Привод_test']
    if trans_name in train_transmission_list:
        df_test.loc[i, trans_name] = 1
df_test[train_transmission_list] = df_test[train_transmission_list].fillna(0)
df_test.drop(['Привод_test'], axis=1, inplace=True)
df_test.head(5)

df_test['Состояние_test'].unique()
df_test.drop(['Состояние_test'], axis=1, inplace=True)
df_test.head(5)

df_test['Владельцы_test'].unique()
df_test['Владельцы_test'] = df_test['Владельцы_test'].apply(lambda cell: cell.replace('\xa0', ' '))
train_owner_quant_list = ['2 владельца', '3 или более']
# Распределение числа владельцев
for i in range(df_test.shape[0]):
    owner_quant = df_test.loc[i, 'Владельцы_test']
    if owner_quant in train_owner_quant_list:
        df_test.loc[i, owner_quant] = 1
df_test.head(5)
df_test[train_owner_quant_list] = df_test[train_owner_quant_list].fillna(0)
df_test.drop(['Владельцы_test'], axis=1, inplace=True)

df_test['ПТС_test'].unique()
# Если Оригинал=1, Дубликат=0
df_test.loc[df_test['ПТС_test']=='Оригинал', 'ПТС_test'] = 1
df_test.loc[df_test['ПТС_test']=='Дубликат', 'ПТС_test'] = 0
df_test['Оригинал ПТС'] = df_test['ПТС_test']
df_test.head(5)
df_test.drop(['ПТС_test'], axis=1, inplace=True)

# Обратное переименование столбца
df_test = df_test.rename(columns={'id_test': 'id'})
df_test.head(5)

# Список двигателей тренировочной выборки
train_engines_list = ['20d', '20d xDrive', '20i xDrive', '25d', '30d', '318i', '320d xDrive', 
                      '320i', '320i xDrive', '35i', '40d', '520d', '520d xDrive', '520i', 
                      '528i xDrive', '530d xDrive', '730Ld xDrive', 'M50d', 'otherEngines']
df_test['engine_test'].unique()
# Распределение двигателей - учитывается то, что двигатель '116i' представлен всеми нулями
for i in range(df_test.shape[0]):
    engine_name = df_test.loc[i, 'engine_test']
    if engine_name in train_engines_list:
        df_test.loc[i, engine_name] = 1
    elif engine_name not in train_engines_list and engine_name != '116i':
        df_test.loc[i, 'otherEngines'] = 1
# Замена нулями отсутствующих параметров и удаление столбца
df_test[train_engines_list] = df_test[train_engines_list].fillna(0)
df_test.drop(['engine_test'], axis=1, inplace=True)
df_test.head(20)

# Заполнение столбцов с полиномиальными признаками
df_test['mileage^2'] = df_test['mileage']**2
df_test['mileage enginePower'] = df_test['mileage'] * df_test['enginePower']
df_test['mileage carAge'] = df_test['mileage'] * df_test['carAge']
df_test['enginePower^2'] = df_test['enginePower']**2
df_test['enginePower carAge'] = df_test['enginePower'] * df_test['carAge']
df_test['carAge^2'] = df_test['carAge']**2
df_test['1/carAge'] = 1/df_test['carAge']
# Удаление столбца carAge
df_test.drop(['carAge'], axis=1, inplace=True)

# Исключая id
test_df_columns = df_test.columns[1:]
test_df_columns
test_df_columns == train_df_columns
len(test_df_columns) == len(train_df_columns)
# Проверка незаполненных значений
df_test.isna().sum()

with open(r'.\data\03_0_kaggle_test_df_transform.csv', mode='w', encoding='utf8') as file:
    file_writer = csv.writer(file, delimiter=';')
    file_writer.writerow(df_test.columns)
    for ii in range(df_test.shape[0]):
        file_writer.writerow(df_test.loc[ii])

RFRModel = RandomForestRegressor(n_estimators=50, max_depth=25)
RFRModel.fit(X, Y)

# Столбцы обучающей выборки
X.columns
df_test = pd.read_csv(r'.\data\03_0_kaggle_test_df_transform.csv', sep=';', encoding='utf8')
df_test
# Удаление столбца, проверка размера
X_test = df_test.drop(['id'], axis=1)
X_test.shape
# Прогноз логарифма цены
y_test_pred = RFRModel.predict(X_test)
# Обработка результата прогноза - преобразование в DataFrame
price_series = pd.Series(y_test_pred, name='price')
test_price_df = pd.DataFrame((df_test['id'], price_series)).T
# Столбец id - целые, стоимость - убираем логарифм
test_price_df['id'] = test_price_df['id'].astype(int)
test_price_df['price'] = test_price_df['price'].apply(lambda cell: np.exp(cell) - 1)
test_price_df
fig = plt.figure(figsize=(7, 5))
axes = fig.add_axes([0, 0, 1, 1])
axes.hist(test_price_df['price'], bins=50)

axes.tick_params(axis='both', labelsize=15)

axes.set_title('Гистограмма цены для тестовой выборки', fontsize=20)
axes.set_xlabel('Стоимость', fontsize=15)
axes.grid(True)
# Результат прогноза
test_price_df.to_csv('.\data\03_13_kaggle_price_predict.csv', sep=',',  index = False)
