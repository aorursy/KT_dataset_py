import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Загрузим датасет
restaurants = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/main_task.csv')
# Переименуем столбцы в питоновском стиле: вместо пробелов подчёркивания, все буквы в нижнем регистре.

restaurants.columns = ['Restaurant_id', 'City', 'Cuisine', 'Ranking', 'Rating',
       'Price_Range', 'Reviews_Number', 'Reviews', 'URL_TA', 'ID_TA']

new_columns = []
for item in restaurants.columns:
    new_columns.append(item.lower())
    
restaurants.columns = new_columns
restaurants.columns
restaurants['restaurant_id'] = restaurants['restaurant_id'].apply(lambda x: x[3:])
restaurants['restaurant_id'] = restaurants['restaurant_id'].astype('int')
restaurants['city'] = restaurants['city'].astype('category')
# Создаём признак ID города
city_list = sorted( list(restaurants['city'].unique()) )
city_id_dict = dict(zip(city_list, np.arange(0, len(city_list), 1)))
restaurants['city_id'] = restaurants['city'].map(city_id_dict)
restaurants['city_id'] = restaurants['city_id'].astype('category')
# Создаём признак является ли город столицей
city_is_capital_dict = {'Amsterdam' : 1, 
                 'Athens' : 1, 
                 'Barcelona' : 0,
                 'Berlin' : 1, 
                 'Bratislava' : 1, 
                 'Brussels' : 1, 
                 'Budapest' : 1, 
                 'Copenhagen' : 1, 
                 'Dublin' : 1, 
                 'Edinburgh' : 0, 
                 'Geneva' : 0, 
                 'Hamburg' : 0, 
                 'Helsinki' : 1, 
                 'Krakow' : 0, 
                 'Lisbon' : 1, 
                 'Ljubljana' : 1, 
                 'London' : 1, 
                 'Luxembourg' : 1, 
                 'Lyon' : 0, 
                 'Madrid' : 1, 
                 'Milan' : 0, 
                 'Munich' : 0, 
                 'Oporto' : 0, 
                 'Oslo' : 1, 
                 'Paris' : 1, 
                 'Prague' : 1, 
                 'Rome' : 1, 
                 'Stockholm' : 1, 
                 'Vienna' : 1, 
                 'Warsaw' : 1, 
                 'Zurich' : 0}

restaurants['is_capital'] = restaurants['city'].map(city_is_capital_dict)
restaurants['is_capital'] = restaurants['is_capital'].astype('category')
# Добавляем данные о населении
city_pop_dict = {'Amsterdam' : 873555, 
                 'Athens' : 664046, 
                 'Barcelona' : 1620343,
                 'Berlin' : 3748148, 
                 'Bratislava' : 432864, 
                 'Brussels' : 1211035, 
                 'Budapest' : 1768073, 
                 'Copenhagen' : 626508, 
                 'Dublin' : 554554, 
                 'Edinburgh' : 518500, 
                 'Geneva' : 201741, 
                 'Hamburg' : 1930996, 
                 'Helsinki' : 648042, 
                 'Krakow' : 774839, 
                 'Lisbon' : 506654, 
                 'Ljubljana' : 292988, 
                 'London' : 9126366, 
                 'Luxembourg' : 116323, 
                 'Lyon' : 515695, 
                 'Madrid' : 3223334, 
                 'Milan' : 1405879, 
                 'Munich' : 1456039, 
                 'Oporto' : 287591, 
                 'Oslo' : 693491, 
                 'Paris' : 2140526, 
                 'Prague' : 1308632, 
                 'Rome' : 2857321, 
                 'Stockholm' : 974073, 
                 'Vienna' : 1899055, 
                 'Warsaw' : 1802237, 
                 'Zurich' : 415215}
restaurants['city_pop'] = restaurants['city'].map(city_pop_dict)
restaurants['city_pop'] = restaurants['city_pop'].astype('int')
# restaurants['city_pop']
# Добавляем данные о странах
country_dict = {'Amsterdam' : 'FRA', 
                 'Athens' : 'GRE', 
                 'Barcelona' : 'SPA', 
                 'Berlin' : 'GER', 
                 'Bratislava' : 'SVK', 
                 'Brussels' : 'BEL', 
                 'Budapest' : 'HUN', 
                 'Copenhagen' : 'DEN', 
                 'Dublin' : 'IRL', 
                 'Edinburgh' : 'GBR', 
                 'Geneva' : 'CHE', 
                 'Hamburg' : 'GER', 
                 'Helsinki' : 'FIN', 
                 'Krakow' : 'POL', 
                 'Lisbon' : 'POR', 
                 'Ljubljana' : 'SVN', 
                 'London' : 'GBR', 
                 'Luxembourg' : 'LUX', 
                 'Lyon' : 'FRA', 
                 'Madrid' : 'SPA', 
                 'Milan' : 'ITA', 
                 'Munich' : 'GER', 
                 'Oporto' : 'POR', 
                 'Oslo' : 'NOR', 
                 'Paris' : 'FRA', 
                 'Prague' : 'CZE', 
                 'Rome' : 'ITA', 
                 'Stockholm' : 'SWE', 
                 'Vienna' : 'AUT', 
                 'Warsaw' : 'POL', 
                 'Zurich' : 'CHE'}
restaurants['country'] = restaurants['city'].map(country_dict)
restaurants['country'] = restaurants['country'].astype('category')
# restaurants['country']
# Переводим данные о стране в числовой показатель (ID)
country_list = sorted( list(restaurants['country'].unique()) )
country_id_dict = dict(zip(country_list, np.arange(0, len(city_list), 1)))
country_id_dict

restaurants['country_id'] = restaurants['country'].map(country_id_dict)
restaurants['country_id'] = restaurants['country_id'].astype('int')
# Пропуски в кухнях заполняем 'Unknown'
restaurants['cuisine'] = restaurants['cuisine'].fillna('Unknown')
# Превратим данные о кухнях из строки в список
def string_parser(x, separator=',', list_to_replace=['']):
    for entry in list_to_replace:
        x = x.replace(entry, '')
    return x.split(sep=separator)


restaurants['cuisine'] = restaurants['cuisine'].apply(string_parser, list_to_replace=["\'", "\"", "[", "]"])
# restaurants['cuisine']
# Создадим новый признак: число кухонь в ресторане
restaurants['cuisine_number'] = restaurants['cuisine'].apply(len)
restaurants['rating'].value_counts()
restaurants['ranking'] = restaurants['ranking'].astype('int')
#restaurants['rating'] = restaurants['rating'].astype('int')
# Заменим пропуски в количестве отзывов на 0
restaurants['reviews_number'] = restaurants['reviews_number'].fillna(0)
restaurants['reviews_number'] = restaurants['reviews_number'].astype('int')
# Отобразим признак price_range в числовой тип price_type

price_range_map = {'$': 1, '$$ - $$$': 100, '$$$$': 1000}
restaurants['price_type'] = restaurants['price_range'].map(price_range_map, na_action='ignore')
restaurants['price_type'] = restaurants['price_type'].fillna(0)
restaurants['price_type'] = restaurants['price_type'].astype('int')
restaurants['price_type'] = restaurants['price_type'].astype('category')
restaurants['reviews'] = restaurants['reviews'].apply(string_parser, separator='],', list_to_replace=[''])

def colunms_from_list(x, ind=0):
    return x[ind]
restaurants['reviews_texts'] = restaurants['reviews'].apply(colunms_from_list, ind=0)
restaurants['reviews_texts'] = restaurants['reviews_texts'].apply(string_parser, list_to_replace=["\'", "\"", "[", "]"])

restaurants['reviews_dates'] = restaurants['reviews'].apply(colunms_from_list, ind=1)
restaurants['reviews_dates'] = restaurants['reviews_dates'].apply(string_parser, list_to_replace=["\'", "\"", "[", "]"])

def from_strlist_to_timelist(x):
    y = []
    if ( len(x)>1 or len(x[0])>1 ):
        try:
            for item in x:
                y.append( pd.to_datetime(item) )
        except: 
            pass
            # print('oops...')            
    else:
        y.append(pd.to_datetime(np.nan))
    return y

restaurants['reviews_dates'] = restaurants['reviews_dates'].apply(from_strlist_to_timelist)

# Создадим новый признак - дату последнего отзыва

def latest_date(x):
    if len(x) > 0:
        return max(x)

restaurants['newest_review_date'] = restaurants['reviews_dates'].apply(latest_date)
# Создадим новый признак - время между первым и последним отзывом (в днях)

def date_interval(x):
    if len(x) > 0:
        return max(x)-min(x)

restaurants['longest_review_interval'] = restaurants['reviews_dates'].apply(date_interval)
restaurants['longest_review_interval'] = restaurants['longest_review_interval'].dt.days
# restaurants
# Дату последнего отзыва превратим в число - сколько дней прошло с этого момета до сегодняшнего дня (в днях)

def from_today(x):
    return pd.datetime.today() - x

restaurants['last_review'] = restaurants['newest_review_date'].apply(from_today)
restaurants['last_review'] = restaurants['last_review'].dt.days
df_restaurant = restaurants[['restaurant_id', 'city_id', 'is_capital', 'city_pop', 'country_id', 
                            'cuisine_number', 'price_type',
                            'ranking', 'rating', 'reviews_number', 'longest_review_interval']]
df_restaurant.info(memory_usage='deep')
df_restaurant = restaurants[['restaurant_id', 'city_id', 'is_capital', 'city_pop', 'country_id', 
                            'cuisine_number', 'price_type',
                            'ranking', 'rating', 'reviews_number', 'longest_review_interval']]
df_restaurant.info(memory_usage='deep')
# Для чистоты выкинем строки с пропущенными данными. Замена их на что-то есть изменение статистики.
# Хотя замена на среднее в среднем запросто может улучшить показатели средней ошибки.
df_restaurant = df_restaurant.dropna()
# Создадим обучающий и тестовой наборы:
df_restaurant_train, df_restaurant_test = train_test_split(df_restaurant, test_size=0.25, random_state=12345)
# Выделим переменные и целевой признак:
train_features = df_restaurant_train.drop(['rating'], axis=1)
train_target = df_restaurant_train['rating']

valid_features = df_restaurant_test.drop(['rating'], axis=1)
valid_target = df_restaurant_test['rating']
### Заготовки для цикла

ns_estimators = []

forest_maes = []
mae_best = len(train_target)
n_estim_best_mae = 1

forest_maes_rounded = []
mae_best_rounded = len(train_target)
n_estim_best_mae_rounded = 1

### Here we go!

# print('=== Random Forest Regressor ===\n')
for n_estim in range(1, 101):
    ns_estimators.append(n_estim)
    
    ### Создаём модель    
    model = RandomForestRegressor(random_state=12345, n_estimators=n_estim)
    model.fit(train_features, train_target)
    valid_predictions = model.predict((valid_features))
    valid_predictions_rounded = np.around(2*valid_predictions)/2
    
    ### Считаем MAE 
    mae = metrics.mean_absolute_error(valid_target, valid_predictions)
    mae_rounded = metrics.mean_absolute_error(valid_target, valid_predictions_rounded)
    forest_maes.append(mae)
    forest_maes_rounded.append(mae_rounded)
    print("\nFor n_estimators = {}: MAE = {:.6f}, MAE_rounded = {:.6f}".format(n_estim, mae, mae_rounded))
    
    ### Отбираем лучшие MAE
    if mae_rounded < mae_best_rounded:
        mae_best_rounded = mae_rounded
        n_estim_best_mae_rounded = n_estim
        best_model_forest_mae_rounded = model
        print("  ! Best MAE for n_estimators {} is {:.6f} (rounded)".format(n_estim_best_mae_rounded, mae_best_rounded))
    if mae < mae_best:
        mae_best = mae
        n_estim_best_mae = n_estim
        best_model_forest_mae = model
        print("    !! Best MAE for n_estimators {} is {:.6f}".format(n_estim_best_mae, mae_best))
df_RandomForestClassifier = pd.DataFrame(data=np.array([forest_maes, forest_maes_rounded]).T, 
                                        index=ns_estimators, columns=['MAE', 'MAE_rounded'])

sns.set_style('darkgrid')
marker_size = 8

fig, ax = plt.subplots(figsize=(15, 10))

plt.plot(df_RandomForestClassifier.index, df_RandomForestClassifier['MAE'], '-rx', linewidth=2, label="Случайный лес, MAE")
plt.plot(df_RandomForestClassifier.index, df_RandomForestClassifier['MAE_rounded'], '--bx', linewidth=2, 
         label="Случайный лес, MAE_rounded")
plt.plot(n_estim_best_mae, mae_best, 'ro', markersize=marker_size)
plt.plot(n_estim_best_mae_rounded, mae_best_rounded, 'bo', markersize=marker_size)
plt.title('Зависимость MAE от параметров моделей')
#plt.xlabel('год выпуска')
    
ax.legend()

print( 'Минимальная MAE для Случайного леса = {:.4f} при числе стволов {}'
      .format(mae_best, n_estim_best_mae ))
print( 'Минимальная MAE (rounded) для Случайного леса = {:.4f} при числе стволов {}'
      .format(mae_best_rounded, n_estim_best_mae_rounded ))