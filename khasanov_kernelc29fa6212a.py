# ======================================================================
# 4. Выбираем авто выгодно. Итоговая работа

# Задача - создать модель, которая будет предсказывать стоимость автомобиля по его характеристикам
# ======================================================================

# Обучающий датасет отсутствует, его нужно найти самостоятельно.
# Шаблон решения имеется: https://www.kaggle.com/itslek/baseline-sf-dst-car-price-prediction-v14
# второй вариант решения: https://www.kaggle.com/annagrechina/baseline-sf-dst-car-price-prediction-v14


# ======================================================================
# 1. Поиск необходимых датасетов с помощью Yandex и Google
# ======================================================================

# Результаты:
# Датасет с сайта auto.ru:
# https://www.kaggle.com/macsunmood/autoru-parsed-0603-1304
# ======================================================================
# Остальные датасеты (не использованы, в качестве найденных результатов):
# https://www.kaggle.com/austinreese/craigslist-carstrucks-data
# https://www.kaggle.com/macsunmood/autoru-parsed-0603-1304 - Auto.ru
# ======================================================================
# Набор данных по ценам на автомобили для автомобильного рынка Азербайджана:
# https://github.com/Sunuba/car_prices_dataset
# ======================================================================
# Cars Prices with Multiple Linear Regression and RFE:
# https://www.kaggle.com/dronax/car-prices-dataset
# https://www.kaggle.com/goyalshalini93/car-data
# https://www.kaggle.com/jenisam/logistic-regression-carsprice
# ======================================================================
# https://www.kaggle.com/sivaganga/car-priceprediction-using-corr
# https://www.kaggle.com/subhashinavolu/linear-regression-cardataset
# https://www.kaggle.com/dronax/car-prices-dataset
# https://www.kaggle.com/jshih7/car-price-prediction?select=ypred_test.csv
# https://www.kaggle.com/avikasliwal/used-cars-price-prediction?select=train-data.csv
# ======================================================================
# ======================================================================
# Обучающий датасет "df_train" - new_data_99_06_03_13_04.csv
# Тестовый датасет "df_test"   - test.csv
# Образец файла в правильном формате "df_sample" - sample_submission.csv
# ======================================================================
# ======================================================================
# 2. Импортируем необходимые библиотеки
# ======================================================================
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from tqdm.notebook import tqdm
from catboost import CatBoostRegressor
import re

# ======================================================================
# 3. Выполним первичный анализ данных
# ======================================================================

# Установим настроечные пар-ры датасета
pd.set_option('display.max_rows', 50) # показывать больше строк
pd.set_option('display.max_columns', 50) # показывать больше колонок

# Загрузим датасет df_train и посмотрим на него
df_train = pd.read_csv('/kaggle/input/autoru-parsed-0603-1304/new_data_99_06_03_13_04.csv', low_memory = False)

df_train.head(5)

# Загрузим датасет df_sample и посмотрим на него позже
df_sample = pd.read_csv('/kaggle/input/sample-submission/sample_submission.csv', low_memory = False)
# Посмотрим на основные характеристики датасета "df_train"
print(df_train.info())
print(df_train.shape)

# ======================================================================
# Краткие выводы №1:
# В датасете содержится 22 столбца и 10980 строк: 8 числовых, 1 булевый и 13 строковых столбцов
# Строковые признаки необходимо будет преобразовать в категориальные переменные;
# Вещественные признаки преобразуем в целочисленные.
# Признак "Unnamed: 0" - можно и вовсе удалить, он "погоды не играет"
# Целевая переменная - "Price"
# ======================================================================

# Переименуем названия столбцов в датафрейме
df_train.columns = ['id', 'body_type', 'brand', 'color', 'fuel_type', 'model_date', 'name', 'number_of_doors', 'production_date',
                    'vehicle_transmission', 'engine_displacement', 'engine_power', 'description', 'mileage', 'components',
                   'type_drive', 'steering_wheel', 'owners', 'tech_passport', 'customs', 'ownership', 'price']

# Поскольку id - это столбец индексов, его можно сразу удалить
df_train.drop(['id'], inplace = True, axis = 1)

# Удалим также столбцы - description, ownership, name
df_train.drop(['description'], inplace = True, axis = 1)
df_train.drop(['ownership'], inplace = True, axis = 1)
df_train.drop(['name'], inplace = True, axis = 1)

# Поскольку в столбцах body_type, model_date, name, production_date, vehicle_configuration, engine_displacement,
# engine _power, type_drive, owners, tech_passport мало пропусков, то эти строки можно также просто удалить
df_train.dropna(inplace = True)

# ======================================================================
# Краткие выводы №2:
# Признаки 'body_type', 'brand', 'color' преобразуем в числовые коды
# ======================================================================
# Посмотрим на столбец body_type и сколько типов кузовов содержит наш датасет
print(pd.DataFrame(df_train.body_type.value_counts())[:15])

print("Значений, встретившихся в столбце более 1000 раз:", (df_train.body_type.value_counts() > 1000).sum())
df_train.loc[:, ['body_type']].info()

# Отобразим конкретные названия типов кузовов, которые встречаются более 1000 раз
print(pd.DataFrame(df_train.body_type.value_counts() > 1000).head(10))

# Посмотрим на распределения в числовых столбцах
# Напишем для этого сразу компактную функцию
def print_numeric_priznak(name_column, text_print):
    print('Распределение ' + str(text_print))
    display(df_train[name_column].value_counts())
    display(df_train[name_column].hist())
    display(df_train[name_column].describe())
    print("Пропущенных значений в столбце " + str(name_column) + ":", df_train[name_column].isnull().sum())
    print('=' * 80)
    
# Преобразуем строковые данные в поле 'body_type' в числовые коды
df_train['body_type'] = df_train['body_type'].apply(lambda x: int(x.replace('Внедорожник 5 дв.', '1')) if x == 'Внедорожник 5 дв.'
                                                    else int(x.replace('Седан', '2')) if x == 'Седан'
                                                    else int(x.replace('Хэтчбек 5 дв.', '3')) if x == 'Хэтчбек 5 дв.'
                                                    else int(x.replace('Лифтбек', '4')) if x == 'Лифтбек'
                                                    else int(x.replace('Универсал 5 дв.', '5')) if x == 'Универсал 5 дв.' 
                                                    else int(x.replace('Седан Long', '6')) if x == 'Седан Long'
                                                    else int(x.replace('Купе', '7')) if x == 'Купе'
                                                    else int(x.replace('Минивэн', '8')) if x == 'Минивэн'
                                                    else int(x.replace('Хэтчбек 3 дв.', '9')) if x == 'Хэтчбек 3 дв.'
                                                    else int(x.replace('Компактвэн', '10')) if x == 'Компактвэн' else int(11))

# Взглянем на распределение body_type
print_numeric_priznak('body_type', 'видов кузовов:')
# Посмотрим на столбец brand и сколько разновидностей марок автомобилей содержит наш датасет
print(pd.DataFrame(df_train.brand.value_counts()))

# Преобразуем строковые данные в поле 'brand' в числовые коды
df_train['brand'] = df_train['brand'].apply(lambda x: int(x.replace('MERCEDES', '1')) if x == 'MERCEDES'
                                            else int(x.replace('VOLKSWAGEN', '2')) if x == 'VOLKSWAGEN'
                                            else int(x.replace('BMW', '3')) if x == 'BMW'
                                            else int(x.replace('NISSAN', '4')) if x == 'NISSAN'
                                            else int(x.replace('TOYOTA', '5')) if x == 'TOYOTA' 
                                            else int(x.replace('AUDI', '6')) if x == 'AUDI'
                                            else int(x.replace('MITSUBISHI', '7')) if x == 'MITSUBISHI'
                                            else int(x.replace('SKODA', '8')) if x == 'SKODA'
                                            else int(x.replace('VOLVO', '9')) if x == 'VOLVO'
                                            else int(x.replace('HONDA', '10')) if x == 'HONDA'
                                            else int(x.replace('LEXUS', '11')) if x == 'LEXUS'
                                            else int(x.replace('INFINITI', '12')) if x == 'INFINITI'
                                            else int(x.replace('SUZUKI', '13')) if x == 'SUZUKI'
                                            else int(0))

# Посмотрим на столбец color и сколько разновидностей цветов содержит наш датасет
print(pd.DataFrame(df_train.color.value_counts()))

# Цвета с сайта https://colorscheme.ru/color-converter.html и https://colorscheme.ru/color-names.html и 
# https://hysy.org/color/:
# 040001 - черный (Чёрный)
# FAFBFB - белый (≈бледный серо-циановый)
# CACECB - серебристый (Гридеперлевый) (≈светло-серый)
# 97948F - серый (Серый Крайола) (≈серый)
# 0000CC - синий (Синий)  (≈тёмно-синий (CSS: #00c))
# 200204 - корица (в техпаспорте авто прописывается, как «Коричневый (металлик)»)
# EE1D19 - красный (Яркий красно-оранжевый) (≈красный)
# 007F00 - зелёный (Зеленый) (≈сумеречный зелёный)
# C49648 - бежевый (Умеренный зеленовато-желтый) (≈апельсиновый)
# 22A0F8 - голубой (Защитно-синий) (≈лазурный)
# 660099 - фиолетовый (Темный пурпурно-фиолетовый) (Тёмный пурпурно-фиолетовый)
# DEA522 - бежевый или светлокоричневый (Золотисто-березовый) (≈янтарный)
# 4A2197 - тёмно-синий или фиолетовый (Персидский индиго) (≈тёмный фиолетово-синий)
# FFD600 - жёлтый (золотистый) (≈золотой) 
# FF8649 - оранжевый (Огненный оранжевый) (≈светло-оранжевый)
# FFC0CB - розовый (Розовый)

# Преобразуем строковые данные в поле 'color' в числовые коды. 
df_train['color'] = df_train['color'].apply(lambda x: int(x.replace('040001', '1')) if x == '040001'
                                            else int(x.replace('FAFBFB', '2')) if x == 'FAFBFB'
                                            else int(x.replace('CACECB', '3')) if x == 'CACECB'
                                            else int(x.replace('97948F', '4')) if x == '97948F'
                                            else int(x.replace('0000CC', '5')) if x == '0000CC' 
                                            else int(x.replace('200204', '6')) if x == '200204'
                                            else int(x.replace('EE1D19', '7')) if x == 'EE1D19'
                                            else int(x.replace('007F00', '8')) if x == '007F00'
                                            else int(x.replace('C49648', '9')) if x == 'C49648'
                                            else int(x.replace('22A0F8', '10')) if x == '22A0F8' 
                                            else int(x.replace('660099', '11')) if x == '660099'
                                            else int(x.replace('DEA522', '12')) if x == 'DEA522'
                                            else int(x.replace('4A2197', '13')) if x == '4A2197'
                                            else int(x.replace('FFD600', '14')) if x == 'FFD600'
                                            else int(x.replace('FF8649', '15')) if x == 'FF8649'
                                            else int(x.replace('FFC0CB', '16')) if x == 'FFC0CB'
                                            else int(0))

# Посмотрим на столбец fuel_type и сколько разновидностей автомобилей по типу двигателей содержит наш датасет
print(pd.DataFrame(df_train.fuel_type.value_counts()))

# Преобразуем строковые данные в поле 'fuel_type' в числовые коды. 
df_train['fuel_type'] = df_train['fuel_type'].apply(lambda x: int(x.replace('бензин', '1')) if x == 'бензин'
                                                    else int(x.replace('дизель', '2')) if x == 'дизель'
                                                    else int(x.replace('гибрид', '3')) if x == 'гибрид'
                                                    else int(x.replace('электро', '4')) if x == 'электро'
                                                    else int(x.replace('газ', '5')) if x == 'газ' else int(0))

# Преобразуем числовые значения в столбце 'model_date' в тип дата
df_train['model_date'] = df_train['model_date'].apply(lambda x: int(x))
# df_train['model_date'] = df_train['model_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y'))

# Преобразуем числовые значения в столбце 'production_date' в тип дата
# df_train['production_date'] = df_train['production_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y'))

# Отобразим различные названия и значения объёмов двигателя
print('Список уникальных значений в столбце engine_displacement', df_train.engine_displacement.unique())
# Функция для корректировки значений объёма двигателя
def set_value_engine_displacement(x):
    if pd.isnull(x):
        return None
    
    if x == 'nan':
        return None
    
    
    #========== Отфильтруем мусорные данные ===========
    if x == '30' or x == '35' or x == '36' or x == '40' or x == '43' or x == '45' or x == '50' or x == '53' or x == '200h' or x == '200t':
        return None    
    
    if x == '30kWh' or x == 'J30' or x == '55' or x == '63' or x == '65' or x == '76' or x == '78' or x == '400h' or x == '400':
        return None
    #=================================================
    
    #================ марка Infinity ===============  
    val = ['EX25', 'EX35', 'EX37', 'FX30d', 'FX35', 'FX37', 'FX45', 'FX50', 'G25', 'G35', 'G37', 'G20',
          'M35', 'M37', 'M56', 'M25', 'M45', 'Q45', 'JX35']
    if x in val:
        x = x.replace('EX', '')
        x = x.replace('FX', '')
        x = x.replace('JX', '')
        x = x.replace('d', '')
        x = x.replace('G', '')
        x = x.replace('M', '')
        x = x.replace('Q', '')
        x = int(x)
        x = x * 100
        x = str(x)
    #================================================
    if x == '350L':
        x = x.replace('L', '')
        x = int(x)
        x = x * 10
        x = str(x)
    #=================== марка BMW ==================
    val = ['M760Li', '760Li', '750Li', '745Li', '740Li', '730Li' '735Li', '728Li']
    if x in val:
        x = x.replace('Li', '')
        x = x.replace('M', '')
        x = int(x)
        x = x - 700
        x = x * 100
        x = str(x)
    
    val = ['M550d', 'M550i', '750d', '730d', '750Ld', '730Ld', '740Ld', '640i', '635i', '630d', '740d', '840i', '840d', '35i', 
           '30i', '30d', '35d', 'M40i', 'M40d', 'M50d', '25d', '35is', '23i', '23d', '40i', '40d']
    if x in val:
        x = int(3000)
        x = str(x)    

    val = ['750i', '745i', '650i', '645i', 'M850i', '850i', 'M50i', '50i', '48i']
    if x in val:
        x = int(4400)
        x = str(x) 
        
    val = ['725Ld', '730i', '620d', '725d', '30i', '18i', '18d', '20i', '25i', '20d', '28i', 'I30', '123d']
    if x in val:
        x = int(2000)
        x = str(x) 

    val = ['728i', '735i', '733i', '760i', '740i']
    if x in val:
        x = x.replace('i', '')
        x = int(x)
        x = x - 700
        x = x * 100
        x = str(x)
    
    val = ['116i', '118i', '120i', '120d', '118d', 'M135i', '135i', '125i', 'M140i', '130i']
    if x in val:
        x = x.replace('i', '')
        x = x.replace('d', '')
        x = x.replace('M', '')
        x = int(x)
        x = x - 100
        x = x * 100
        x = str(x)  
    
    val = ['220i', '220d', 'M235i', '218i', '216d', '218d']
    if x in val:        
        x = x.replace('i', '')
        x = x.replace('d', '')
        x = x.replace('M', '')
        x = int(x)
        x = x - 200
        x = x * 100
        x = str(x)  
    
    val = ['320d', '320i', '325xi', '325i', '316i', '318i', '328i', '330i', 'M340i', '335i', '316', '323i',
           '318d', '330xi', '330d', '320xd', '335xi', '335d', '340i']
    if x in val:
        x = x.replace('xi', '')
        x = x.replace('xd', '')
        x = x.replace('i', '')
        x = x.replace('d', '')
        x = x.replace('M', '')
        x = int(x)
        x = x - 300
        x = x * 100
        x = str(x) 
    
    val = ['sDrive18i', 'xDrive20d', 'xDrive18d', 'xDrive20i', 'sDrive20i']
    if x in val:
        x = x.replace('sDrive', '')
        x = x.replace('xDrive', '')
        x = x.replace('i', '')
        x = x.replace('d', '')
        x = int(x)
        x = x * 100
        x = str(x)     
    #================================================    
    val = ['Long', '7S-tronic', 'tiptronic', 'S-tronic', 'Tiptronic', '8tiptronic', 'clean', 'V10', 'V8', 'CS', 'N42', 
           'N46', 'Electro', 'Competition', 'ActiveHybrid', 'Sport', 'Type', 'US', 'AMT', 'Hybrid', 'Shuttle', 'del', 
           'Spike', 'Spada', 'Arctic', 'S', 'R', 'All-Terrain', 'Pullman', 'GT', 'Marco', 'L3', 'L2', 'L1', 'VR-4', 
           'Grandis', 'CZ3', 'CZ1', 'CZT', 'Plus', 'CZ2', 'Cargo', 'Easy', 'Super', 'Classic', 'Cedia', '5-speed', 
           'PHEV', 'Cubic', '30kWh', 'Latio', '100', 'Scout', 'DSG7', 'Green', 'DSG', 'ACT', 'Crescent', 'Landy', 
           'XL-7', 'Wide', 'Gracia', 'Japan', 'Prominent', 'Fielder', 'i', 'Ceres', 'Runx', 'Premio', 'SF', 'Lucida', 
           'Emina', '79', 'Cygnus', 'Noah', 'Qualis', 'Plug-in', 'JDM', 'CrewMax', 'Double', 'Ardeo', 'Multimode', 
           'full-time', 'part-time', 'Maxi', 'BiFuel', '4MOTION', 'KAT', 'TGI', 'DSG-6', 'Alltrack', 'EcoFuel', 
           '5-Seater', '4-Seater', 'Cross', 'Allspace', 'XC', 'XX', 'Solio', 'J30', 'GTi', 'e-Golf', 'RS', 'Axio', 
           'GT-Four', 'ultra', '16V', '8V', 'S4', '190', '94Ah']
    if x in val:
        return None
    
    x = x.replace('sDrive', '')
    x = x.replace('xDrive', '')
    x = x.replace('hyb', '')
#     x = x.replace('EX', '')
    x = x.replace('FX', '')
    
    x = x.replace('xi', '')
    x = x.replace('si', '')
    x = x.replace('is', '')
    x = x.replace('xd', '')
    x = x.replace('LI', '')
    
    x = x.replace('JX', '')
    x = x.replace('Ld', '')
    x = x.replace('E', '')
    x = x.replace('N', '')
#     x = x.replace('M', '')
    
    x = x.replace('d', '')
    x = x.replace('i', '')
    x = x.replace('I', '')
    x = x.replace('G', '')
    x = x.replace('М', '')
    
    x = x.replace('G', '')
    x = x.replace('Q', '')
    x = x.replace('h', '')
    x = x.replace('J', '')
    x = x.replace('L', '')
    
    x = x.replace('C', '')
    x = x.replace('e', '')
    x = x.replace('A', '')
    x = x.replace('t', '')
    x = x.replace('L', '')
    x = x.replace('s', '')
    
    #================================================
    val = ['140', '150', '160', '170', '180', '200', '220', '230', '240', '250', '260', '270', '280', '290', 
           '300', '320', '350', '380', '400', '450', '460', '470', '500', '560', '570']
    if x in val:
        x = int(x)
        x = x * 10
        x = str(x) 
    #================================================ 
          
    if x == '':
        return None    
   
    if x != x:
        x = float(x)

    if x != None:
        x = float(x)
        
    if x >= 0.6 and x <= 6.3:
        x = x * 1000
        x = int(x)

    if x >= 105 and x <= 126:
        x = (x - 100) * 100
        x = int(x)
    
    if x >= 315 and x <= 330:
        x = (x - 300) * 100
        x = int(x)
    
    if x >= 420 and x <= 440:
        x = (x - 400) * 100
        x = int(x)
    
    if x >= 518 and x <= 550:
        x = (x - 500) * 100
        x = int(x)

    if x >= 620 and x <= 650:
        x = (x - 600) * 100
        x = int(x)
    
    if x >= 730 and x <= 745:
        x = (x - 700) * 100
        x = int(x)       
        
    return int(x)
# Преобразуем вещественные в целочисленные значения в столбце 'number_of_doors'
df_train['number_of_doors'] = df_train['number_of_doors'].apply(lambda x: int(x))

# Преобразуем данные в столбце 'engine_displacement' в числовой тип
df_train['engine_displacement'] = df_train['engine_displacement'].apply(set_value_engine_displacement)

# Преобразуем строковые данные в поле 'vehicle_transmission' в числовые коды. 
df_train['vehicle_transmission'] = df_train['vehicle_transmission'].apply(lambda x: int(x.replace('MECHANICAL', '1')) if x == 'MECHANICAL'
                                                                          else int(x.replace('AUTOMATIC', '2')) if x == 'AUTOMATIC'
                                                                          else int(x.replace('ROBOT', '3')) if x == 'ROBOT'
                                                                          else int(x.replace('VARIATOR', '4')) if x == 'VARIATOR'
                                                                          else int(0))

# Преобразуем вещественные значения в целочисленные в столбце engine_power
df_train['engine_power'] = df_train['engine_power'].apply(lambda x: int(x) if x != None else x)

# Преобразуем строковые данные в поле 'type_drive' в числовые коды
df_train['type_drive'] = df_train['type_drive'].apply(lambda x: int(x.replace('передний', '1')) if x == 'передний'
                                                      else int(x.replace('полный', '2')) if x == 'полный'
                                                      else int(x.replace('задний', '3')) if x == 'задний'
                                                      else int(0))

# Преобразуем строковые данные в поле 'steering_wheel' в числовые коды
df_train['steering_wheel'] = df_train['steering_wheel'].apply(lambda x: int(x.replace('LEFT', '1')) if x == 'LEFT' else int(2))

# Преобразуем вещественные значения в целочисленные в столбце owners
df_train['owners'] = df_train['owners'].apply(lambda x: int(x) if x != None else x)

# Преобразуем строковые данные в поле 'tech_passport' в числовые коды
df_train['tech_passport'] = df_train['tech_passport'].apply(lambda x: int(x.replace('ORIGINAL', '1')) if x == 'ORIGINAL' else int(2))
# ======================================================================
# 2.2. Выполним первичный анализ данных 'test.csv'
# ======================================================================

# Установим настроечные пар-ры датасета
pd.set_option('display.max_rows', 50) # показывать больше строк
pd.set_option('display.max_columns', 50) # показывать больше колонок

# Загрузим датасет df_test и посмотрим на него
df_test = pd.read_csv('/kaggle/input/dataset-test/test.csv', low_memory = False)
df_test.head(5)
# Посмотрим на основные характеристики датасета "df_test"
print(df_test.info())
print(df_test.shape)

# ======================================================================
# Краткие выводы №2.1:
# В датасете содержится 23 столбца и 3837 строк: 5 числовых и 18 строковых столбцов
# Строковые признаки необходимо будет преобразовать в числовые переменные;
# Вещественные признаки преобразуем в целочисленные.
# Признак "id" - можно и вовсе удалить, он "погоды не играет"
# Целевая переменная - отсутствует
# Необходимо синхронизация названия, порядка и кодировки значений датасета "df_test" с "df_train"
# ======================================================================

# Переименуем названия столбцов в датафрейме
df_test.columns = ['body_type', 'brand', 'color', 'fuel_type', 'model_date', 'name', 'number_of_doors', 'production_date',
                    'body_type_english', 'vehicle_transmission', 'engine_displacement', 'engine_power', 'description', 
                   'mileage', 'components', 'type_drive', 'steering_wheel', 'condition', 'owners', 'tech_passport', 
                   'customs', 'ownership', 'id']

# Удалим также столбцы - name, body_type_english, description, components, condition, ownership 
df_test.drop(['name'], inplace = True, axis = 1)
df_test.drop(['body_type_english'], inplace = True, axis = 1)
df_test.drop(['description'], inplace = True, axis = 1)
# df_test.drop(['components'], inplace = True, axis = 1)
df_test.drop(['condition'], inplace = True, axis = 1)
df_test.drop(['ownership'], inplace = True, axis = 1)

# Поскольку id - это столбец индексов, его можно сразу удалить
df_test.drop(['id'], inplace = True, axis = 1)
# Посмотрим на столбец body_type и сколько типов кузовов содержит наш датасет
print(pd.DataFrame(df_test.body_type.value_counts()))

# Преобразуем строковые данные в поле 'body_type' в числовые коды
df_test['body_type'] = df_test['body_type'].apply(lambda x: int(x.replace('внедорожник 5 дв.', '1')) if x == 'внедорожник 5 дв.'
                                                  else int(x.replace('седан', '2')) if x == 'седан'
                                                  else int(x.replace('хэтчбек 5 дв.', '3')) if x == 'хэтчбек 5 дв.'
                                                  else int(x.replace('лифтбек', '4')) if x == 'лифтбек'
                                                  else int(x.replace('универсал 5 дв.', '5')) if x == 'универсал 5 дв.' 
                                                  else int(x.replace('седан Long', '6')) if x == 'седан Long'
                                                  else int(x.replace('купе', '7')) if x == 'купе'
                                                  else int(x.replace('минивэн', '8')) if x == 'минивэн'
                                                  else int(x.replace('хэтчбек 3 дв.', '9')) if x == 'хэтчбек 3 дв.'
                                                  else int(x.replace('компактвэн', '10')) if x == 'компактвэн' else int(11))

# Посмотрим на столбец brand и сколько разновидностей марок автомобилей содержит наш датасет
print(pd.DataFrame(df_test.brand.value_counts()))

# Преобразуем строковые данные в поле 'brand' в числовые коды
df_test['brand'] = df_test['brand'].apply(lambda x: int(x.replace('MERCEDES', '1')) if x == 'MERCEDES'
                                          else int(x.replace('VOLKSWAGEN', '2')) if x == 'VOLKSWAGEN'
                                          else int(x.replace('BMW', '3')) if x == 'BMW'
                                          else int(x.replace('NISSAN', '4')) if x == 'NISSAN'
                                          else int(x.replace('TOYOTA', '5')) if x == 'TOYOTA' 
                                          else int(x.replace('AUDI', '6')) if x == 'AUDI'
                                          else int(x.replace('MITSUBISHI', '7')) if x == 'MITSUBISHI'
                                          else int(x.replace('SKODA', '8')) if x == 'SKODA'
                                          else int(x.replace('VOLVO', '9')) if x == 'VOLVO'
                                          else int(x.replace('HONDA', '10')) if x == 'HONDA'
                                          else int(x.replace('LEXUS', '11')) if x == 'LEXUS'
                                          else int(x.replace('INFINITI', '12')) if x == 'INFINITI'
                                          else int(x.replace('SUZUKI', '13')) if x == 'SUZUKI'
                                          else int(0))

# Посмотрим на столбец color и сколько разновидностей цветов содержит наш датасет
print(pd.DataFrame(df_test.color.value_counts()))

# Цвета с сайта https://colorscheme.ru/color-converter.html и https://colorscheme.ru/color-names.html и 
# https://hysy.org/color/:
# 040001 - черный (Чёрный)
# FAFBFB - белый (≈бледный серо-циановый)
# CACECB - серебристый (Гридеперлевый) (≈светло-серый)
# 97948F - серый (Серый Крайола) (≈серый)
# 0000CC - синий (Синий)  (≈тёмно-синий (CSS: #00c))
# 200204 - корица (в техпаспорте авто прописывается, как «Коричневый (металлик)»)
# EE1D19 - красный (Яркий красно-оранжевый) (≈красный)
# 007F00 - зелёный (Зеленый) (≈сумеречный зелёный)
# C49648 - бежевый (Умеренный зеленовато-желтый) (≈апельсиновый)
# 22A0F8 - голубой (Защитно-синий) (≈лазурный)
# 660099 - фиолетовый (Темный пурпурно-фиолетовый) (Тёмный пурпурно-фиолетовый)
# DEA522 - бежевый или светлокоричневый (Золотисто-березовый) (≈янтарный)
# 4A2197 - тёмно-синий или фиолетовый (Персидский индиго) (≈тёмный фиолетово-синий)
# FFD600 - жёлтый (золотистый) (≈золотой) 
# FF8649 - оранжевый (Огненный оранжевый) (≈светло-оранжевый)
# FFC0CB - розовый (Розовый)

# Преобразуем строковые данные в поле 'color' в числовые коды. 
df_test['color'] = df_test['color'].apply(lambda x: int(x.replace('чёрный', '1')) if x == 'чёрный'
                                          else int(x.replace('белый', '2')) if x == 'белый'
                                          else int(x.replace('серебристый', '3')) if x == 'серебристый'
                                          else int(x.replace('серый', '4')) if x == 'серый'
                                          else int(x.replace('синий', '5')) if x == 'синий' 
                                          else int(x.replace('коричневый', '6')) if x == 'коричневый'
                                          else int(x.replace('красный', '7')) if x == 'красный'
                                          else int(x.replace('зелёный', '8')) if x == 'зелёный'
                                          else int(x.replace('бежевый', '9')) if x == 'бежевый'
                                          else int(x.replace('голубой', '10')) if x == 'голубой' 
                                          else int(x.replace('пурпурный', '11')) if x == 'пурпурный'
                                          else int(x.replace('жёлтый', '12')) if x == 'жёлтый'
                                          else int(x.replace('фиолетовый', '13')) if x == 'фиолетовый'
                                          else int(x.replace('золотистый', '14')) if x == 'золотистый'
                                          else int(x.replace('оранжевый', '15')) if x == 'оранжевый'
                                          else int(x.replace('розовый', '16')) if x == 'розовый'
                                          else int(0))

# Посмотрим на столбец fuel_type и сколько разновидностей двигателей наш датасет
print(pd.DataFrame(df_test.fuel_type.value_counts()))

# Преобразуем строковые данные в поле 'fuel_type' в числовые коды. 
df_test['fuel_type'] = df_test['fuel_type'].apply(lambda x: int(x.replace('бензин', '1')) if x == 'бензин'
                                                  else int(x.replace('дизель', '2')) if x == 'дизель'
                                                  else int(x.replace('гибрид', '3')) if x == 'гибрид'
                                                  else int(x.replace('электро', '4')) if x == 'электро'
                                                  else int(x.replace('газ', '5')) if x == 'газ' else int(0))

# Преобразуем числовые значения в столбце 'model_date' в тип дата
df_test['model_date'] = df_test['model_date'].apply(lambda x: int(x))
# df_test['model_date'] = df_test['model_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y'))

# Преобразуем вещественные в целочисленные значения в столбце 'number_of_doors'
df_test['number_of_doors'] = df_test['number_of_doors'].apply(lambda x: int(x))

# Преобразуем данные в столбце 'production_date' в числовой тип
df_test['production_date'] = df_test['production_date'].apply(lambda x: int(x))
# df_test['production_date'] = df_test['production_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y'))

# Преобразуем строковые данные в поле 'vehicle_transmission' в числовые коды. 
df_test['vehicle_transmission'] = df_test['vehicle_transmission'].apply(lambda x: int(x.replace('механическая', '1')) if x == 'механическая'
                                                                        else int(x.replace('автоматическая', '2')) if x == 'автоматическая'
                                                                        else int(x.replace('роботизированная', '3')) if x == 'роботизированная'
                                                                        else int(x.replace('вариатор', '4')) if x == 'вариатор'
                                                                        else int(0))
# Отобразим различные названия и значения объёмов двигателя
print('Список уникальных значений в столбце engine_displacement', df_test.engine_displacement.unique())

# Отобразим различные названия и значения в поле type_drive
print('Список уникальных значений в столбце type_drive', df_test.type_drive.unique())

# Отобразим различные названия и значения в поле steering_wheel
print('Список уникальных значений в столбце steering_wheel', df_test.steering_wheel.unique())

# Отобразим различные названия и значения в поле owners
print('Список уникальных значений в столбце owners', df_test.owners.unique())

# Отобразим различные названия и значения в поле customs
print('Список уникальных значений в столбце customs', df_test.customs.unique())
# Напишем функцию для корректировки значений объёмов двигателя
def set_value_engine_displacement(x):
    if pd.isnull(x):
        return None
    
    if x == 'nan':
        return None
    
    #========== Отфильтруем мусорные данные ===========
    x = x.replace(' LTR', '')
    
    if x == 'undefined':
        return None
    #==================================================
    
    # Преобразуем значения в кубические сантиметры
    if x != None:
        x = float(x)
        x = x * 1000
    
    return int(x)

# Напишем функцию для корректировки значений кол-ва лошадиных сил
def set_value_engine_power(x):
    if pd.isnull(x):
        return None
    
    if x == 'nan':
        return None
    
    #========== Отфильтруем мусорные данные ===========
    x = x.replace(' N12', '')
    #==================================================    
    # Преобразуем в числовые значения
    if x != None:
        x = int(x)
    
    return int(x)

# Напишем функцию для корректировки значений кол-ва владельцев
def set_value_owners(x):
    if pd.isnull(x):
        return None
    
    if x == 'nan':
        return None
    
    #========== Отфильтруем мусорные данные ===========
    x = x.replace('\xa0владельца', '')
    x = x.replace('\xa0владелец', '')
    x = x.replace(' или более', '')
    #==================================================    
    # Преобразуем в числовые значения
    if x != None:
        x = int(x)
    
    return int(x)
# Преобразуем данные в столбце 'engine_displacement' в числовой тип
df_test['engine_displacement'] = df_test['engine_displacement'].apply(set_value_engine_displacement)

# Преобразуем данные в столбце 'engine_power' в числовой тип
df_test['engine_power'] = df_test['engine_power'].apply(set_value_engine_power)

# Преобразуем вещественные значения в целочисленные в столбце mileage
df_test['mileage'] = df_test['mileage'].apply(lambda x: int(x) if x != None else x)

# Преобразуем строковые данные в поле 'type_drive' в числовые коды
df_test['type_drive'] = df_test['type_drive'].apply(lambda x: int(x.replace('передний', '1')) if x == 'передний'
                                                    else int(x.replace('полный', '2')) if x == 'полный'
                                                    else int(x.replace('задний', '3')) if x == 'задний'
                                                    else int(0))

# Преобразуем строковые данные в поле 'steering_wheel' в числовые коды
df_test['steering_wheel'] = df_test['steering_wheel'].apply(lambda x: int(x.replace('Левый', '1')) if x == 'Левый' else int(2))

# Преобразуем данные в столбце 'owners' в числовой тип
df_test['owners'] = df_test['owners'].apply(set_value_owners)

# Преобразуем строковые данные в поле 'tech_passport' в числовые коды
df_test['tech_passport'] = df_test['tech_passport'].apply(lambda x: int(x.replace('Оригинал', '1')) if x == 'Оригинал' else int(2))

# Преобразуем строковые данные в поле 'customs' в булевые значения
df_test['customs'] = df_test['customs'].apply(lambda x: bool(x.replace('Растаможен', 'True')) if x == 'Растаможен' else bool('False'))
# Отобразим различные названия и значения объёмов двигателя
print('Список уникальных значений в столбце components', df_test.components.unique())
# Распарсим столбец components в df_train
def parse_train_components(value):
    if value != value or pd.isnull(value):
        return None
    
    result = value.replace(' ', '')
    result = result.replace("'", '')
    
    first_bracket = result.find('[')
    second_bracket = result.find(']')
    
    if first_bracket != -1 and second_bracket != -1:
        return result[first_bracket+1:second_bracket].split(',')
    else:
        return []

df_train['components'] = df_train['components'].apply(parse_train_components)

# Распарсим столбец components в df_test
import json

def parse_test_complectation(value):
    if value != value or pd.isnull(value):
        return []
    
    value = value.replace("['",'')
    value = value.replace("']",'')
    
    # переведём с английского на русский значения
    result = value.replace('Антиблокировочная система (ABS)', 'abs')
    result = value.replace('Подушка безопасности водителя', 'airbag-driver')
    result = value.replace('Подушка безопасности пассажира', 'airbag-passenger')
    result = value.replace('Электростеклоподъёмники передние', 'electro-window-front')
    result = value.replace('Центральный замок', 'lock')
    
    result = value.replace('Регулировка руля по вылету', 'wheel-configuration1')
    result = value.replace('Регулировка руля по высоте', 'wheel-configuration2')
    result = value.replace('Иммобилайзер', 'immo')
    result = value.replace('Электропривод зеркал', 'electro-mirrors')
    result = value.replace('Электрообогрев боковых зеркал', 'mirrors-heat')
    
    result = value.replace('Аудиоподготовка', 'audiopreparation')    
    result = value.replace('Подогрев передних сидений', 'front-seats-heat')
    result = value.replace('Складывающееся заднее сиденье', 'seat-transformation')
    result = value.replace('Климат-контроль 1-зонный', 'climate-control-1')
    result = value.replace('Кондиционер', 'condition')

    
    result = value.replace('Бортовой компьютер', 'computer')
    result = value.replace('Аудиосистема', 'audiosystem-cd')
    result = value.replace('Электростеклоподъёмники задние', 'electro-window-back')
    result = value.replace('Подушки безопасности боковые', 'airbag-side')
    result = value.replace('Система стабилизации (ESP)', 'esp')
    
    values = json.loads(value)
    result = []
    for item in values:
        if 'values' in item.keys():
            result.extend(item['values'])
    return result

df_test['components'] = df_test['components'].apply(parse_test_complectation)
# ======================================================================
# 3.1 Выполним создание новых признаков на тренировочной и тестовой моделях
# ======================================================================

# соберем все уникальные названия в признаке комплектации
from collections import defaultdict
complectation = defaultdict(int)
for value in df_train['components']:
    if value == value and value != None:
        for item in value:
            complectation[item] += 1
            
# всего получилось 155 уникальных значений, для генерации столбцов оставим 20 самых часто встречающихся
most_frequent_num = 20
sort_complectation = sorted(complectation.items(), key = lambda x: x[1], reverse=True)
most_frequent_items = []
for item in sort_complectation[:most_frequent_num]:
    most_frequent_items.append(item[0])

def fill_components_item(value):
    if item in value:
        return 1
    else:
        return 0
    
for item in most_frequent_items:
    df_train[item] = df_train['components'].apply(fill_components_item)
# соберем все уникальные названия в признаке комплектации
from collections import defaultdict
complectation_test = defaultdict(int)
for value in df_test['components']:
    if value == value and value != None:
        for item in value:
            complectation_test[item] += 1

most_frequent_num = 20
sort_complectation = sorted(complectation.items(), key = lambda x: x[1], reverse=True)
most_frequent_items = []
for item in sort_complectation[:most_frequent_num]:
    most_frequent_items.append(item[0])

def fill_components_item(value):
    if item in value:
        return 1
    else:
        return 0
    
for item in most_frequent_items:
    df_test[item] = df_test['components'].apply(fill_components_item)
df_train.head(5)
df_test.head(5)
# ======================================================================
# Data Preprocessing
# ======================================================================
def preproc_data(df_input):
    df_output = df_input.copy()    
   
    # убираем не нужные для модели признаки
    df_output.drop(['components'], axis = 1, inplace = True)
    # очистим строки, содержащие пропуски
    df_output.dropna(inplace = True)
    
    # Переводим признаки из float в int (иначе catboost выдает ошибку)
    for feature in ['body_type', 'brand', 'color', 'fuel_type', 'model_date', 'number_of_doors',
                    'production_date', 'vehicle_transmission', 'engine_displacement', 'engine_power', 'mileage',
                    'type_drive', 'steering_wheel', 'owners', 'tech_passport', 'customs', 'price',
                    'airbag-driver', 'electro-window-front', 'lock', 'wheel-configuration1', 'immo', 'abs', 
                    'airbag-passenger', 'electro-mirrors', 'mirrors-heat', 'audiopreparation', 'computer', 
                    'wheel-configuration2', 'electro-window-back', 'airbag-side', 'esp', 'front-seats-heat', 
                    'audiosystem-cd', 'seat-transformation', 'climate-control-1', 'condition']:
        df_output[feature] = df_output[feature].astype('int32')
    
        
    return df_output

def preproc_data_modify(df_input):
    df_output = df_input.copy()  

    
    # убираем не нужные для модели признаки
    df_output.drop(['components'], axis = 1, inplace = True)
    # очистим строки, содержащие пропуски
    df_output = df_output.fillna(0)
    
    # Переводим признаки из float в int (иначе catboost выдает ошибку)
    for feature in ['body_type', 'brand', 'color', 'fuel_type', 'model_date', 'number_of_doors',
                    'production_date', 'vehicle_transmission', 'engine_displacement', 'engine_power', 'mileage',
                    'type_drive', 'steering_wheel', 'owners', 'tech_passport', 'customs',
                    'airbag-driver', 'electro-window-front', 'lock', 'wheel-configuration1', 'immo', 'abs', 
                    'airbag-passenger', 'electro-mirrors', 'mirrors-heat', 'audiopreparation', 'computer', 
                    'wheel-configuration2', 'electro-window-back', 'airbag-side', 'esp', 'front-seats-heat', 
                    'audiosystem-cd', 'seat-transformation', 'climate-control-1', 'condition']:
        df_output[feature] = df_output[feature].astype('int32')
    
        
    return df_output
# Используем для наглядности матрицу корреляций:
df_train.corr()

# ======================================================================
# Краткие выводы №2.2:
# В датасете содержатся следующие признаки скоррелированные почти полностью:
# 'airbag-driver' vs 'electro-window-front' vs 'lock' vs 'wheel-configuration-1' vs
# 'immo' vs 'abs' vs 'airbag-passenger' vs 'electro-mirrors' vs 'mirrors-heat' vs
# 'audiopreparation' vs 'computer' vs 'wheel-configuration2' vs 'electro-window-back' vs
# 'airbag-side' vs 'esp' vs 'front-seats-heat' vs 'audiosystem-cd'
# Удалим указанные признаки, кроме первого.
# ======================================================================
# ======================================================================
# 5. Проверим, есть ли статистическая разница в распределении оценок по номинативным признакам, с помощью теста Стьюдента. 
# Проверим нулевую гипотезу о том, что распределения математических оценок по различным параметрам неразличимы:
# ======================================================================
def get_stat_dif(column):
    cols = df_train.loc[:, column].value_counts().index[:5]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df_train.loc[df_train.loc[:, column] == comb[0], 'price'], 
                        df_train.loc[df_train.loc[:, column] == comb[1], 'price']).pvalue \
            <= 0.05/len(combinations_all): # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


for col in ['body_type', 'brand', 'color', 'fuel_type', 'model_date', 'number_of_doors',
            'production_date', 'vehicle_transmission', 'engine_displacement', 'engine_power', 'mileage',
            'type_drive', 'steering_wheel', 'owners', 'tech_passport', 'customs',
            'airbag-driver', 'electro-window-front', 'lock', 'wheel-configuration1', 'immo', 'abs', 'airbag-passenger', 'electro-mirrors', 
            'mirrors-heat', 'audiopreparation', 'computer', 'wheel-configuration2', 'electro-window-back', 'airbag-side', 'esp', 
            'front-seats-heat', 'audiosystem-cd', 'seat-transformation', 'climate-control-1', 'condition']:
    get_stat_dif(col)

# ======================================================================
# 6. Заключение
# ======================================================================

# Для оценки 'важности' признаков на целевую переменную score использовали t-test и основанный на нём p-value для 
# тестирования нулевой гипотезы.
# В итоге значимыми осталось лишь 35 признаков (столбцов, колонок):
# ======================================================================
# 4.1 Выполним обучение тренировочной модели
# ======================================================================
RANDOM_SEED = 42
VAL_SIZE   = 0.1   # 33%
N_FOLDS    = 5
# CATBOOST
ITERATIONS = 6000
LR         = 0.05

train_preproc = preproc_data(df_train)
X = train_preproc.drop(['price'], axis = 1)

y = 0.95 * train_preproc.price.values
cat_features_ids = ['body_type', 'brand', 'color', 'fuel_type', 'model_date', 'number_of_doors',
                    'production_date', 'vehicle_transmission', 'engine_displacement', 'engine_power', 'mileage',
                    'type_drive', 'steering_wheel', 'owners', 'tech_passport', 'customs',
                    'airbag-driver', 'electro-window-front', 'lock', 'wheel-configuration1', 'immo', 'abs', 'airbag-passenger', 'electro-mirrors', 
                    'mirrors-heat', 'audiopreparation', 'computer', 'wheel-configuration2', 'electro-window-back', 'airbag-side', 'esp', 
                    'front-seats-heat', 'audiosystem-cd', 'seat-transformation', 'climate-control-1', 'condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = VAL_SIZE, shuffle = True, random_state = RANDOM_SEED)

# Fit model
model = CatBoostRegressor(iterations = ITERATIONS, learning_rate = LR, random_seed = RANDOM_SEED,
                          eval_metric='MAPE', custom_metric=['R2', 'MAE'])

model.fit(X_train, y_train, cat_features = cat_features_ids, eval_set=(X_test, y_test), verbose_eval = 100,
          use_best_model = True, plot = True)
model.save_model('catboost_single_model_baseline.model')
features_importances = pd.DataFrame(data = model.feature_importances_, index = X.columns, columns = ['FeatImportant'])
features_importances.sort_values(by = 'FeatImportant', ascending = False).head(20)
X_sub = preproc_data_modify(df_test)
# Submission
predict_submission = model.predict(X_sub)
predict_submission
VERSION    = 5

df_sample['price'] = predict_submission
df_sample.to_csv(f'submission_v{VERSION}.csv', index = False)
df_sample.head(10)
# Cross Validation
def cat_model(y_train, X_train, X_test, y_test):
    model = CatBoostRegressor(iterations = ITERATIONS,
                              learning_rate = LR,
                              eval_metric='MAPE',
                              random_seed = RANDOM_SEED,)
    model.fit(X_train, y_train,
              cat_features=cat_features_ids,
              eval_set=(X_test, y_test),
              verbose=False,
              use_best_model=True,
              plot=False)
    
    return(model)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))
submissions = pd.DataFrame(0, columns = ["sub_1"], index = df_sample.index) # куда пишем предикты по каждой модели
score_ls = []
splits = list(KFold(n_splits = N_FOLDS, shuffle = True, random_state = RANDOM_SEED).split(X, y))

for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total = N_FOLDS):
    # use the indexes to extract the folds in the train and validation data
    X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]
    # model for this fold
    model = cat_model(y_train, X_train, X_test, y_test,)
    # score model on test
    test_predict = model.predict(X_test)
    test_score = mape(y_test, test_predict)
    score_ls.append(test_score)
    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")
    # submissions
    submissions[f'sub_{idx+1}'] = model.predict(X_sub)
    model.save_model(f'catboost_fold_{idx+1}.model')
    
print(f'Mean Score: {np.mean(score_ls):0.3f}')
print(f'Std Score: {np.std(score_ls):0.4f}')
print(f'Max Score: {np.max(score_ls):0.3f}')
print(f'Min Score: {np.min(score_ls):0.3f}')
# ======================================================================
# 5.1 Выполним обучение тренировочной модели с удалёнными скоррелированными признаками
# и сравним новые результаты с прошлыми - bestTest = 0.1469561899
# ======================================================================
# Краткие выводы №2.3:
# Выполнил обучение тренировочной модели с удалёнными скоррелированными признаками и сравнил новые результаты с прошлыми 
# Результат практически такой же - 0.1468297, но считает на 6 минут быстрее
# ======================================================================
RANDOM_SEED = 42
VAL_SIZE   = 0.1   # 33%
N_FOLDS    = 5
# CATBOOST
ITERATIONS = 6000
LR         = 0.05

train_preproc = preproc_data(df_train)
X = train_preproc.drop(['price'], axis = 1)

X = X.drop(['electro-window-front'], axis = 1)
X = X.drop(['lock'], axis = 1)
X = X.drop(['wheel-configuration1'], axis = 1)
X = X.drop(['immo'], axis = 1)
X = X.drop(['abs'], axis = 1)

X = X.drop(['airbag-passenger'], axis = 1)
X = X.drop(['electro-mirrors'], axis = 1)
X = X.drop(['mirrors-heat'], axis = 1)
X = X.drop(['audiopreparation'], axis = 1)
X = X.drop(['computer'], axis = 1)

X = X.drop(['wheel-configuration2'], axis = 1)
X = X.drop(['electro-window-back'], axis = 1)
X = X.drop(['airbag-side'], axis = 1)
X = X.drop(['esp'], axis = 1)
X = X.drop(['front-seats-heat'], axis = 1)
X = X.drop(['audiosystem-cd'], axis = 1)

y = 0.95 * train_preproc.price.values
cat_features_ids = ['body_type', 'brand', 'color', 'fuel_type', 'model_date', 'number_of_doors',
                    'production_date', 'vehicle_transmission', 'engine_displacement', 'engine_power', 'mileage',
                    'type_drive', 'steering_wheel', 'owners', 'tech_passport', 'customs',
                    'airbag-driver', 'seat-transformation', 'climate-control-1', 'condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = VAL_SIZE, shuffle = True, random_state = RANDOM_SEED)

# Fit model
model = CatBoostRegressor(iterations = ITERATIONS, learning_rate = LR, random_seed = RANDOM_SEED,
                          eval_metric='MAPE', custom_metric=['R2', 'MAE'])

model.fit(X_train, y_train, cat_features = cat_features_ids, eval_set=(X_test, y_test), verbose_eval = 100,
          use_best_model = True, plot = True)
model.save_model('catboost_single_model_baseline.model')
features_importances = pd.DataFrame(data = model.feature_importances_, index = X.columns, columns = ['FeatImportant'])
features_importances.sort_values(by = 'FeatImportant', ascending = False).head(20)
X_sub = preproc_data_modify(df_test)

X_sub = X_sub.drop(['electro-window-front'], axis = 1)
X_sub = X_sub.drop(['lock'], axis = 1)
X_sub = X_sub.drop(['wheel-configuration1'], axis = 1)
X_sub = X_sub.drop(['immo'], axis = 1)
X_sub = X_sub.drop(['abs'], axis = 1)

X_sub = X_sub.drop(['airbag-passenger'], axis = 1)
X_sub = X_sub.drop(['electro-mirrors'], axis = 1)
X_sub = X_sub.drop(['mirrors-heat'], axis = 1)
X_sub = X_sub.drop(['audiopreparation'], axis = 1)
X_sub = X_sub.drop(['computer'], axis = 1)

X_sub = X_sub.drop(['wheel-configuration2'], axis = 1)
X_sub = X_sub.drop(['electro-window-back'], axis = 1)
X_sub = X_sub.drop(['airbag-side'], axis = 1)
X_sub = X_sub.drop(['esp'], axis = 1)
X_sub = X_sub.drop(['front-seats-heat'], axis = 1)
X_sub = X_sub.drop(['audiosystem-cd'], axis = 1)
# Submission
predict_submission = model.predict(X_sub)
predict_submission
VERSION    = 5

df_sample['price'] = predict_submission
df_sample.to_csv(f'submission_v{VERSION}.csv', index = False)
df_sample.head(10)
# Cross Validation
def cat_model(y_train, X_train, X_test, y_test):
    model = CatBoostRegressor(iterations = ITERATIONS,
                              learning_rate = LR,
                              eval_metric='MAPE',
                              random_seed = RANDOM_SEED,)
    model.fit(X_train, y_train,
              cat_features=cat_features_ids,
              eval_set=(X_test, y_test),
              verbose=False,
              use_best_model=True,
              plot=False)
    
    return(model)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))
submissions = pd.DataFrame(0, columns = ["sub_1"], index = df_sample.index) # куда пишем предикты по каждой модели
score_ls = []
splits = list(KFold(n_splits = N_FOLDS, shuffle = True, random_state = RANDOM_SEED).split(X, y))

for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total = N_FOLDS):
    # use the indexes to extract the folds in the train and validation data
    X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]
    # model for this fold
    model = cat_model(y_train, X_train, X_test, y_test,)
    # score model on test
    test_predict = model.predict(X_test)
    test_score = mape(y_test, test_predict)
    score_ls.append(test_score)
    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")
    # submissions
    submissions[f'sub_{idx+1}'] = model.predict(X_sub)
    model.save_model(f'catboost_fold_{idx+1}.model')
    
print(f'Mean Score: {np.mean(score_ls):0.3f}')
print(f'Std Score: {np.std(score_ls):0.4f}')
print(f'Max Score: {np.max(score_ls):0.3f}')
print(f'Min Score: {np.min(score_ls):0.3f}')

# Выполнил обучение тренировочной модели с удалёнными скоррелированными признаками и сравнил новые результаты с прошлыми - bestTest = 0.1469561899
# Результат практически такой же = 0.1468297, но считает на 6 минут быстрее