# Imports



# Data processing

import pandas as pd



# Date procesing

import datetime



# DataVis

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns
# Загрузка данных

nobel_df = pd.read_csv('../input/nobel-laureates/archive.csv')

nobel_df.head(3)
# Список столбцов

nobel_df.columns
# Для работы со знаками зодиака

# потребуются даты рождения





def srt_do_date(str):

    """Преобразует строку в формат datetime"""

    try:

        dt = pd.to_datetime(str, format='%Y-%m-%d')

    except ValueError:

        dt = 0

    pass

    return dt





nobel_df['Birth Date'] = nobel_df['Birth Date'].apply(srt_do_date)
nobel_df.head()
# Избавимся от строк с некорректными датами рождения

nobel_df = nobel_df[nobel_df['Birth Date'] != 0]
nobel_df.info()
# Удалим строки с пустыми значениями дат рождения

nobel_df = nobel_df.dropna(subset=['Birth Date'])
nobel_df.info()
# Преобразуем интервал дат в знак зодиака





def zodiac(date):

    """Возвращает знак зодиака по дате рождения"""

    

    day = date.day

    month = date.month

    

    if (day >= 21 and day <= 31 and month == 3) or (day >= 1 and day <= 20 and month == 4):

        zodiac = 'Овен'

    elif (day >= 21 and day <= 31 and month == 4) or (day >= 1 and day <= 21 and month == 5):

        zodiac = 'Телец'

    elif (day >= 22 and day <= 31 and month == 5) or (day >= 1 and day <= 21 and month == 6):

        zodiac = 'Близнецы'

    elif (day >= 22 and day <= 31 and month == 6) or (day >= 1 and day <= 22 and month == 7):

        zodiac = 'Рак'

    elif (day >= 23 and day <= 31 and month == 7) or (day >= 1 and day <= 21 and month == 8):

        zodiac = 'Лев'

    elif (day >= 22 and day <= 31 and month == 8) or (day >= 1 and day <= 23 and month == 9):

        zodiac = 'Дева'

    elif (day >= 24 and day <= 31 and month == 9) or (day >= 1 and day <= 23 and month == 10):

        zodiac = 'Весы'

    elif (day >= 24 and day <= 31 and month == 10) or (day >= 1 and day <= 22 and month == 11):

        zodiac = 'Скорпион'

    elif (day >= 23 and day <= 31 and month == 11) or (day >= 1 and day <= 22 and month == 12):

        zodiac = 'Стрелец'

    elif (day >= 23 and day <= 31 and month == 12) or (day >= 1 and day <= 20 and month == 1):

        zodiac = 'Козерог'

    elif (day >= 21 and day <= 31 and month == 1) or (day >= 1 and day <= 19 and month == 2):

        zodiac = 'Водолей'

    else:

        zodiac ='Рыбы'

    

    return zodiac
nobel_df['zodiac'] = nobel_df['Birth Date'].apply(zodiac)
# Посчитаем и отранжируем частоту знаков

nobel_df['zodiac'].value_counts()