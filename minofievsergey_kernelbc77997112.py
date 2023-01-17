#подгружаем необходимые библиотеки
import numpy as np 
import pandas as pd 
import datetime as dt
from datetime import date, timedelta
import re
from statistics import mean
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os
from collections import Counter
#подгружаем данные
df_train = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/main_task.csv')
df_test = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/kaggle_task.csv')
#df_train = pd.read_csv('main_task.csv')
#df_test = pd.read_csv('kaggle_task.csv')
#sample_submission = pd.read_csv('sample_submission.csv')
#Фиксируем random_seed и версии пакетов
!pip freeze > requirements.txt
RANDOM_SEED = 42
# На старте будем работать с одним датасетом. помечаем признаками и объединяем
df_train['sample'] = 1 # трейн
df_test['sample'] = 0 # тест
df_test['Rating'] = 0 # нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями
df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
#df = df.drop(['Name'], axis=1) #Name поле не имеет значения для нашей модели
#Дополнительный справочник о городах участнпиках рейтинга
#Данные взяты из Википедии
#capital - 1 столица, 0 нет
#average_income - уровень среднего дохода в месяц по странам в долларах
#population_density - плотность населения по городам (число жителей, приходящееся на 1 км²)
#population - численность населения
#numberofrestaraunts - количество ресторанов в городе
#country - столица

city_info = {
    'Paris':
    {
        'capital': 1,
        'average_income':3332, 
        'population_density':20781,
        'population': 2148000,
        'numberofrestaraunts': 16684,
        'country': 'France'
    },
    'Stockholm':
    {
        'capital': 1,
        'average_income':2893, 
        'population_density':5139.7,
        'population': 974000,
        'numberofrestaraunts': 2882,
        'country': 'Sweden'
    },
    'London':
    {
        'capital': 1,
        'average_income':2703, 
        'population_density':5667,
        'population': 9126366,
        'numberofrestaraunts': 19374,
        'country': 'England'
    },
    'Berlin':
    {
        'capital': 1,
        'average_income':4392, 
        'population_density':4088,
        'population': 3748148,
        'numberofrestaraunts': 6962,
        'country': 'Germany'
    },
    'Bratislava':
    {
        'capital': 1,
        'average_income':1283, 
        'population_density':1171,
        'population': 432864,
        'numberofrestaraunts': 1201,
        'country': 'Slovakia'
    },
    'Vienna':
    {
        'capital': 1,
        'average_income':2940, 
        'population_density':4502.88,
        'population': 1889000,
        'numberofrestaraunts': 3951,
        'country': 'Austria'
    },
    'Rome':
    {
        'capital': 1,
        'average_income':2726, 
        'population_density':2234,
        'population': 2873000,
        'numberofrestaraunts': 10558,
        'country': 'Italy'
    },
    'Madrid':
    {
        'capital': 1,
        'average_income':2133, 
        'population_density':8653.5,
        'population': 3266126,
        'numberofrestaraunts': 10891,
        'country': 'Spain'
    },
    'Dublin':
    {
        'capital': 1,
        'average_income':3671, 
        'population_density':3689,
        'population': 554554,
        'numberofrestaraunts': 2298,
        'country': 'Ireland'
    },
    'Brussels':
    {
        'capital': 1,
        'average_income':3930, 
        'population_density':5497,
        'population': 1211035,
        'numberofrestaraunts': 3525,
        'country': 'Belgium'
    },
    'Warsaw':
    {
        'capital': 1,
        'average_income':1253, 
        'population_density':3449,
        'population': 1708000,
        'numberofrestaraunts': 3040,
        'country': 'Poland'
    },
    'Budapest':
    {
        'capital': 1,
        'average_income':1187, 
        'population_density':3330.5,
        'population': 1768073,
        'numberofrestaraunts': 2917,
        'country': 'Hungary'
    },
    'Copenhagen':
    {
        'capital': 1,
        'average_income':6192, 
        'population_density':6214.7,
        'population': 626508,
        'numberofrestaraunts': 2329,
        'country': 'Denmark'
    },
    'Amsterdam':
    {
        'capital': 1,
        'average_income':3238, 
        'population_density':4768,
        'population': 873555,
        'numberofrestaraunts': 3847,
        'country': 'The Netherlands'
    },
    'Lisbon':
    {
        'capital': 1,
        'average_income':1288, 
        'population_density':6243.9,
        'population': 506654,
        'numberofrestaraunts': 4682,
        'country': 'Portugal',
    },
    'Prague':
    {
        'capital': 1,
        'average_income':1454, 
        'population_density':2506,
        'population': 1319000,
        'numberofrestaraunts': 5213,
        'country': 'The Czech Republic'
    },
    'Oslo':
    {
        'capital': 1,
        'average_income':5450, 
        'population_density':1483.41,
        'population': 673000,
        'numberofrestaraunts': 1303,
        'country': 'Norway'
    },
    'Helsinki':
    {
        'capital': 1,
        'average_income':3908, 
        'population_density':899,
        'population': 648042,
        'numberofrestaraunts': 1478,
        'country': 'Finland'
    },
    'Edinburgh':
    {
        'capital': 1,
        'average_income':2703, 
        'population_density':4140,
        'population': 518500,
        'numberofrestaraunts': 1852,
        'country': 'Scotland'
    },
    'Ljubljana':
    {
        'capital': 1,
        'average_income':1914, 
        'population_density':1736,
        'population': 292988,
        'numberofrestaraunts': 583,
        'country': 'Slovenia'
    },
    'Athens':
    {
        'capital': 1,
        'average_income':1203, 
        'population_density':7500,
        'population': 664046,
        'numberofrestaraunts': 2441,
        'country': 'Greece'
    },
    'Luxembourg':
    {
        'capital': 1,
        'average_income':5854, 
        'population_density':2240,
        'population': 116323,
        'numberofrestaraunts': 716,
        'country': 'Luxembourg'
    },
        'Munich':
    {
        'capital': 0,
        'average_income':4392, 
        'population_density':4713,
        'population': 1456000,
        'numberofrestaraunts': 3018,
        'country': 'Germany'
    },
    'Oporto':
    {
        'capital': 0,
        'average_income':1288, 
        'population_density':5703,
        'population': 214000,
        'numberofrestaraunts': 1902,
        'country': 'Portugal'
    },
    'Milan':
    {
        'capital': 0,
        'average_income':2726, 
        'population_density':7588.97,
        'population': 1352000,
        'numberofrestaraunts': 7000,
        'country': 'Italy'
    },
    'Barcelona':
    {
        'capital': 0,
        'average_income':2133, 
        'population_density':15779,
        'population': 1620343,
        'numberofrestaraunts': 9309,
        'country': 'Spain'
    },
    'Zurich':
    {
        'capital': 0,
        'average_income':6244, 
        'population_density':4666,
        'population': 402000,
        'numberofrestaraunts': 1792,
        'country': 'Switzerland'
    },
    'Lyon':
    {
        'capital': 0,
        'average_income':3332, 
        'population_density':10041,
        'population': 515695,
        'numberofrestaraunts': 2701,
        'country': 'France'
    },
    'Hamburg':
    {
        'capital': 0,
        'average_income':4392, 
        'population_density':2388.57,
        'population': 1930996,
        'numberofrestaraunts': 3138,
        'country': 'Germany'
    },
    'Geneva':
    {
        'capital': 0,
        'average_income':6244, 
        'population_density':12589,
        'population': 201741,
        'numberofrestaraunts': 1665,
        'country': 'Switzerland'
    },
    'Krakow':
    {
        'capital': 0,
        'average_income':1253, 
        'population_density':2344,
        'population': 774839,
        'numberofrestaraunts': 1621,
        'country': 'Poland'
    }
}
#Добавление признаков на проверку на NaN перед их заменой в колонках
df['number_of_reviews_is_nan'] = pd.isna(df['Number of Reviews']).astype('uint8')
df['price_range_is_nan'] = pd.isna(df['Price Range']).astype('uint8')
#Заполнение пропусков в количестве отзывов
df['Number of Reviews'].fillna(0,inplace=True)
df['Price Range'].fillna('$$ - $$$',inplace=True)
#Добавление относительного ранга ресторана по городам в зависимости от числа ресторанов в нем
#чем больше к единице, тем выше ранг ресторана
df['relative_ranking'] = 1-(df['Ranking'] / df['City'].map(df.groupby(['City'])['Ranking'].max()))
#Замена интервалов цен на 0 (низкая), 1 (средняя) и 2 (высокая), NaN - на 1 (как медианное значение), 
#а также вычисление среднего ценового уровня в разных городах
price_dict = {'$$ - $$$':1,'$':0,'$$$$':2,float('nan'):1}
df['Price'] = df['Price Range'].map(price_dict)
##Создание dummy variables из price_range
price_dummies = pd.get_dummies(df['Price Range'])
df = pd.concat([df, price_dummies], axis = 1)
#Признак столица/не столица
df['capital'] = df['City'].apply(lambda x: city_info[x]['capital'])
df['nocapital'] = df['capital'].apply(lambda x: 1 if x == 0 else 0)
#Признак название страны
df['country_name'] = df['City'].apply(lambda x: city_info[x]['country'])
#Признак уровень среднего дохода в месяц по странам в долларах
df['average_income'] = df['City'].apply(lambda x: city_info[x]['average_income'])
#Признак число жителей, приходящееся на 1 км²
df['population_density'] = df['City'].apply(lambda x: city_info[x]['population_density'])
#Признак население города
df['population'] = df['City'].apply(lambda x: city_info[x]['population'])
#Признак количество ресторанов¶
df['num_restaurants'] = df['City'].apply(lambda x: city_info[x]['numberofrestaraunts'])
#Признак индекс отношения количества ресторанов на душу населения
df['restaraunts_per_people'] = df['City'].apply(lambda x: (city_info[x]['numberofrestaraunts'] / city_info[x]['population']))
#Признак индекс отношения населения на количество ресторанов
df['people_per_restaraunts'] = df['City'].apply(lambda x: (city_info[x]['population'] / city_info[x]['numberofrestaraunts']))
#Признак индекс отношения населения на 1 кв.км на количество ресторанов
df['peopled_per_restaraunts'] = df['City'].apply(lambda x: (city_info[x]['population_density'] / city_info[x]['numberofrestaraunts']))
#Признак относительный рейтинг по городу
df['ranking_per_city'] = df['relative_ranking'] * df['peopled_per_restaraunts']
#Создание dummy variables из городов датасета
city_dummies = pd.get_dummies(df['City'])
df = pd.concat([df, city_dummies], axis = 1)
#Вычисление количества дней между отзывами
import datetime
pattern = re.compile('\d+\W\d+\W\d+')
def choose_date(reviews):
    dates = []
    if type(reviews) is str:
        dates = pattern.findall(reviews)
    return dates

def compute_days_reviews(value):
    rd = 0
    if type(value) is list:
        if len(value) == 2:
            rd = abs((pd.to_datetime(str(value[0]))-pd.to_datetime(str(value[1]))).days)
            return rd #if rd < 3000 else 0 #отсекаем отзывы более 5лет с большой вероятностью как ошибочное 
    return 0

df['dates'] = df['Reviews'].apply(choose_date)
df['days_between_reviews'] = df['dates'].apply(compute_days_reviews)
#Готовим список кухонь и словарь список кухонь в разрезе городов для след. обработки
cuisine_dict = {}
cuisine_list = []
re_cuisine = re.compile(r"'([^']+)'")

def get_cuisines_list(value):
    if type(value) is not str:
        cuisine_list.append('Vegetarian Friendly') #Vegetarian Friendly как медианное значение
        return ['Vegetarian Friendly']
    else:
        cuisine_list.extend(re_cuisine.findall(value))
        return re_cuisine.findall(value)
    
def create_cuisine_dictionary(raw):
    if raw[1] not in cuisine_dict.keys():
        cuisine_dict[raw[1]] = []
    cuisine_dict[raw[1]].extend(raw[2])


df['Cuisine Style'] = df['Cuisine Style'].apply(get_cuisines_list)
temp = df.apply(create_cuisine_dictionary,axis=1)
#Создание признаков из всех типов кухонь в ресторанах
cuisines = pd.Series(cuisine_list).unique()
for cuisine in cuisines:
    df[cuisine] = df['Cuisine Style'].apply(lambda x: 1 if cuisine in x else 0)
#Коэффициент популярности кухонь конкретного ресторана в разрезе городов
def get_cuisines_popularity(raw):
    result = 0
    if type(raw[2]) is list:
        c = Counter(cuisine_dict[raw[1]])
        total_sum = sum(c.values())
        for i in raw[2]:
            result = result + c[i]
    return result/total_sum

df['x_cuisine_popularity'] = df.apply(get_cuisines_popularity,axis=1)
#Вычисление количества кухонь для каждого ресторана
df['cuisine_count'] = df['Cuisine Style'].apply(lambda x: len(x) if type(x) is list else 1)
#Вычисление среднего количества кухонь в ресторанах по городам
df['cuisine_count_mean'] = df['City'].map(df.groupby('City')['cuisine_count'].mean())
#Признак сетевого ресторана
df.columns
df.tail()
df['id'] = df['Restaurant_id'].apply(lambda x: int(x[3:]))
id_restaurants = pd.Series(df['id'].value_counts())
chain_restaurants = id_restaurants[id_restaurants > 1]
df['chain_restaurants'] = df['id'].apply(lambda x: 1 if x in chain_restaurants else 0)
df['not_chain_restaurants'] = df['chain_restaurants'].apply(lambda x: 1 if x == 0 else 0)
#Признак потенциально возможных затрат на ресторан 
#df['cost_per_restaraunt'] =  df['Ranking']*df['average_income']*df['population'] / df['num_restaurants']
#Признак относительный рейтинг по городу от доходов в разрезе населения / на единицу ресторанов
df['ranking_per_idxppr'] = df['relative_ranking']*(df['x_cuisine_popularity']/df['num_restaurants']) #(1 - 1/(1+df['cost_per_restaraunt']))*(1-1/(1+df['x_cuisine_popularity']))
#Относительный Признак потенциально возможных затрат на ресторан с учетом его популярности
#df['x_cost_per_restaraunt'] = df['cost_per_restaraunt'] * df['x_cuisine_popularity'] * (1 + 10*df['chain_restaurants'])
df_output = df.drop(['Restaurant_id', 'City', 'Cuisine Style', 'Reviews', 'Price Range', 'URL_TA', 'ID_TA', 'id', 'dates','country_name'], axis=1)
# Теперь выделим тестовую часть
train_data = df_output.query('sample == 1').drop(['sample'], axis=1)
test_data = df_output.query('sample == 0').drop(['sample'], axis=1)

y = train_data.Rating.values            # наш таргет
X = train_data.drop(['Rating'], axis=1)
# Загружаем специальный удобный инструмент для разделения датасета:
from sklearn.model_selection import train_test_split
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию (параметр test_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# проверяем
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)
def round_output(result):
    if result <= 0.25:
        return 0
    elif 0.25 < result <= 0.75:
        return 0.5
    elif 0.75 < result <= 1.25:
        return 1
    elif 1.25 < result <= 1.75:
        return 1.5
    elif 1.75 < result <= 2.25:
        return 2
    elif 2.25 < result <= 2.75:
        return 2.5
    elif 2.75 < result <= 3.25:
        return 3
    elif 3.25 < result <= 3.75:
        return 3.5
    elif 3.75 < result <= 4.25:
        return 4
    elif 4.25 < result <= 4.75:
        return 4.5
    else:
        return 5
    
for i in range(y_pred.size):
    y_pred[i] = round_output(y_pred[i])
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
#Submission
test_data = test_data.drop(['Rating'], axis=1)
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission['Rating'] = sample_submission['Rating'].apply(round_output)
sample_submission.to_csv('submission_new_forked.csv', index=False)
sample_submission.to_csv('solution.csv', index = False)  
sample_submission.head(10)
















































