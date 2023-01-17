# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, timedelta 

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

# Загружаем специальный удобный инструмент для разделения датасета:
from sklearn.model_selection import train_test_split

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
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
fig, ax = plt.subplots(figsize=(15, 5))
sns_heatmap = sns.heatmap(
    data.isnull(), yticklabels=False, cbar=False)
df_train.head(5)
df_test.info()
df_test.head(5)
sample_submission.head(5)
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
# Столбцы с пропусками
for col in df_train.columns:
    none_val_count = len(df_train) - df_train[col].isna().value_counts()[0]
    if none_val_count !=0:
        print(col, none_val_count)
# Неявные пропуски в столбце 'Reviews', например data['Reviews'][0]
print(data['Reviews'][0])  
print(data['Reviews'][1])
# Какие признаки можно считать категориальными?
data.nunique(dropna=False)
# фиксируем параметры графиков
sns.set(rc={'figure.figsize':(7,5)}, font_scale=0.5, style='whitegrid')
def drow_boxplot(data, column):
    """Функция упрощает отрисовку boxplot для анализируемого столбца и столбца score """
    sns.set(rc={'figure.figsize':(7,5)}, font_scale=1.5, style='whitegrid')
    sns.boxplot(x=data[column])


def drow_hist_joinplot(dt, column, column_bins):
    """Функция отрисовывает гистограмму и jointplot для столбца column
     в интервале (perc25 - 1.5 * IQR, perc75 +  1.5 * IQR) и для столбща столбца 'Rating'"""
   
    median = dt[column].median()
    IQR = dt[column].quantile(0.75) - dt[column].quantile(0.25)
    perc25 = dt[column].quantile(0.25)
    perc75 = dt[column].quantile(0.75)

    if perc75 != perc25:
        dt[column][dt[column].between(perc25 - 1.5 * IQR, perc75 + 1.5 * IQR)].hist(bins=column_bins, label='IQR')
        plt.legend()
        sns.jointplot(x=column, y='Rating', data=dt[dt[column] < perc75 + 1.5 * IQR], kind='reg')
    else:
        dt[column].hist(bins=column_bins, label='All data')
        plt.legend()
        sns.jointplot(x=column, y='Rating', data=dt)

        
def print_data_col_info(data, column_a, column_b):
    """Функция выводит информацию о количестве выбросов, 
    пустых значений, коэфф. корреляции, ..."""

    display(pd.DataFrame(data[column_a].value_counts()))
    median = data[column_a].quantile(0.5)
    IQR = data[column_a].quantile(0.75) - data[column_a].quantile(0.25)
    perc25 = data[column_a].quantile(0.25)
    perc75 = data[column_a].quantile(0.75)      
    print('Информация по столбцу - '+ column_a+ ':')
    if column_a != column_b:
        Kcor = round(
            data[[column_a, column_b]].corr(method='pearson')[column_a][1], 3)
    else:
        Kcor = 1
    nan = int(len(data) -
              data[column_a].describe()[0])  # количество 'nan' значений

    left_border = len(data[data[column_a] < perc25 - 1.5 * IQR])
    right_border = len(data[data[column_a] > perc75 + 1.5 * IQR])

    outliers = left_border + right_border

    print(
        " 25-й перцентиль: {},\n".format(perc25),
        "75-й перцентиль: {},\n".format(perc75),
        "Медиана: {},\n".format(median), "IQR: {},\n".format(IQR),
        "Количество 'nan' значений: {},\n".format(nan),
        "Количество выбросов: {}, слева: {}, справа: {}\n".format(
            outliers, left_border, right_border),
        "Границы выбросов: [{f}, {l}],\n".format(f=perc25 - 1.5 * IQR,
                                                 l=perc75 + 1.5 * IQR),
        "Kоэф. корреляции между \'{}\' и \'{}\' : {},\n".format(
            column_a, column_b, Kcor))


def drow_nomin_hist(data, column, column_beans):
    """Функция отрисовыват гистограмму для номинативных признаков"""
    
    data[column].hist(bins=column_beans)


def print_nomin_info(data, column):   
    """Функция выводит информацию о номинативном признаке """
    
    data[column] = data[column].astype(str).apply(lambda x: None if x.strip() == '' else x)
    display(pd.DataFrame(data[column].value_counts()))
    print("Значений, встретившихся в столбце более 10 раз:",
          (data[column].value_counts() > 10).sum())
    print("Уникальных значений:", data[column].nunique())
    print(data.loc[:, [column]].info())
    data[column].value_counts(ascending=True).plot(kind='bar')
drow_boxplot(data, 'Ranking')
print_data_col_info(data, 'Ranking', 'Rating' )
drow_hist_joinplot(data, 'Ranking', 100)
# Распределение ранга в Лондоне
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
print_nomin_info(data, 'City')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['Ranking'][df_train['City'] == x].hist(bins=50)
plt.show()
data['Number of Reviews'].fillna(0.0, inplace = True)
data['Number of Reviews'] = data['Number of Reviews'].apply(lambda x: int(x))
drow_boxplot(data, 'Number of Reviews')
drow_hist_joinplot(data, 'Number of Reviews', 100)
print_nomin_info(data, 'Rating')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
data['id_ta'] = data['ID_TA'].apply(lambda x: int(x[1:]))
data['id_ta'].isna().value_counts()
drow_boxplot(data, 'id_ta')
data['id_ta'].hist(bins=100)
drow_hist_joinplot(data, 'id_ta', 100)
plt.rcParams['figure.figsize'] = (15,10)
sns.heatmap(data.drop(['sample'], axis=1).corr(),)
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
# Заполняем пропуски в столбце 'Number of Reviews'
data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...
data['Number of Reviews'].fillna(0, inplace=True)
data['Number of Reviews_2'] = data['Number of Reviews']**2
data['Price Range'].fillna('$$ - $$$', inplace=True)


print(data['Price Range'].value_counts())

def price_range_one_hot_encod(price_range):
        ''' Замена строковых значений числовыми '''

        if price_range == '$$ - $$$':
            return 1
        if price_range == '$$$$':
            return 2
        if price_range == '$$':
            return 3

        
data['price_range'] = data['Price Range'].apply(price_range_one_hot_encod)


# Заполняем пропуски в столбце 'Number of Reviews'
data['Price Range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')

data = pd.get_dummies(data, columns = ['price_range'], dummy_na=True)  
data = data.drop(columns = ['Price Range'])
#  Словать с населением городов
population_dict = {'London' :    8250000, 
                   'Paris' :     2041826, 
                   'Madrid' :    3142880, 
                   'Barcelona' : 1590000, 
                   'Berlin' :    3350000, 
                   'Milan' :     1342000, 
                   'Rome' :      2800000, 
                   'Prague' :    1294000, 
                   'Vienna' :    1840000, 
                   'Amsterdam' : 860000, 
                   'Brussels' :  179000, 
                   'Hamburg' :   1841000, 
                   'Munich' :    1471000, 
                   'Lyon' :      516000, 
                   'Stockholm' : 975000, 
                   'Budapest' :  1762000, 
                   'Warsaw' :    1810000, 
                   'Dublin' :    1173000, 
                   'Copenhagen': 974000, 
                   'Athens' :    665000, 
                   'Edinburgh' : 513000, 
                   'Lisbon' :    506000, 
                   'Zurich' :    434000, 
                   'Oporto' :    237000, 
                   'Geneva' :    201000, 
                   'Krakow' :    779000, 
                   'Oslo' :      693000, 
                   'Helsinki' :  643000, 
                   'Bratislava': 413000, 
                   'Luxembourg': 626000, 
                   'Ljubljana' : 284000,
                   }

data['population'] = data.apply(lambda x: population_dict[x['City']], axis = 1)
data['rest_counts'] = data['City'].apply(lambda x: data['City'].value_counts()[x])
data['rest_density'] = data['rest_counts'] / data['population']
city_rest_count = data.groupby(['City'])['Ranking'].max()


data['city_rest_count'] = data['City'].apply(lambda x: city_rest_count[x])
data['city_rest_density'] = data['population']/data['Ranking']
    
    

data['norm_ranking'] = data.groupby('City')['Ranking'].transform(lambda x: (x-x.mean())/x.std())

data['rel_ranking'] = (data['Ranking'] / data['city_rest_count'])
# добавление полиномиального признака
# data['ranking_reviews_count'] = data['Ranking'] * data['Number of Reviews']   # на проверочных данных увеличивае МАЕ, на текущих уменьшает
# top_city_list = list(data['City'].value_counts().index[:8])
# def replace_city(city):
#     '''Функция возвращает значение 'Other city', если входного параметра нет в топ n городов '''
    
#     if city in top_city_list:
#         return city
#     return 'Other city'
# data['City'] = data['City'].apply(replace_city)
data['reviews_in_city'] = data['City'].apply(lambda x: data.groupby(['City'])['Number of Reviews'].sum().sort_values(ascending=False)[x])
data['relative_rank_reviews'] = data['Ranking'] / data['reviews_in_city']
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data['Cuisine Style'] = data['Cuisine Style'].fillna("['Vegetarian Friendly', 'European']") # Vegetarian Friendly  ниже выявлены самые популярные виды кухонь
data['Cuisine Style'].value_counts()
# Определяем наиболее популярные виды кухни 
cuisine_styles = []
for cuisine_style in data['Cuisine Style'].value_counts().index:
    cuisine_style_list = cuisine_style[2:-2].split("', '")
    for cuisine_style_item in cuisine_style_list:
        cuisine_styles.append(cuisine_style_item)

        
cuisine_styles = set(cuisine_styles)

cuisine_styles_dict = {}
new_cuisine_styles_dict = {}

for cuisine in cuisine_styles:
    cuisine_styles_dict[cuisine] = 0

for i in range(len(data)):   
    for cuisine_style in data['Cuisine Style'].iloc[i][2:-2].split("', '"):
        cuisine_styles_dict[cuisine_style]+=1
        
for key,value in cuisine_styles_dict.items():
#     if value > 3000:
     new_cuisine_styles_dict[key] = value
     print(key,value)  
# Добавим столбец с количнством кухонь

def get_cuisine_style_count(cuisine_styles):
     '''Функция возвращает количество кухонь в реторане'''
     
     return len(cuisine_styles[1:-1].split("', '"))
    
data['Cuisine_styles_count'] = data['Cuisine Style'].apply(get_cuisine_style_count)       
# data['ddd'] = data['norm_ranking'] * data['Cuisine_styles_count']   # в данной выборке MAE уменьшает, в проверочной увеличивает
# # Dummy переменные для 'Cuisine Style'    # Увеличивает МАЕ
# for  cuisine in cuisine_styles:
#     data[cuisine] = 0
# for i in range(len(data)):
#     for cuisine in cuisine_styles:
#         if cuisine in data.iloc[i]['Cuisine Style']:
#              data.at[i, cuisine] = 1
def get_reviews(reviews):
    '''Функция возвращает список с отзывами'''
    return reviews.split("], [")[0]

def get_reviews_dates(reviews):
    '''Функция возвращает списокс датами'''
    return reviews.split("], [")[1]

data['Reviews'].fillna('[[], []] ', inplace = True)
data['reviews'] = data['Reviews'].apply(get_reviews)
data['reviews'] = data['reviews'].replace('', 'Nan')


data['reviews_dates'] = data['Reviews'].apply(get_reviews_dates)


def get_first_date(reviews_dates):
    '''Функция возвращает первую дату из списка'''
    
    date = reviews_dates.split(", ")[0]
    if date == ']]':
        return None
    if "']" in date[1:-1]:
        return    datetime.strptime(date[1:-3], '%m/%d/%Y').timestamp()
    if "]" == date[1:-1]:
        return    None
    return  datetime.strptime(date[1:-1], '%m/%d/%Y').timestamp()

def get_second_date(reviews_dates):
    '''Функция возвращает вторую дату из списка'''
    if len(reviews_dates.split(", ")) == 1:
        return None
    if len(reviews_dates.split(", ")) > 1:
        date = reviews_dates.split(", ")[1]
        if "']" in date[1:-1]:
            return    datetime.strptime(date[1:-3], '%m/%d/%Y').timestamp()
        return datetime.strptime(date[1:-3], '%m/%d/%Y').timestamp()

data['first_date'] = data.reviews_dates.apply(get_first_date)
data['first_date'] = data['first_date'].fillna(data['first_date'][data['first_date'] != 0.0].min())


data['second_date'] = data.reviews_dates.apply(get_second_date)
data['second_date'] = data['second_date'].fillna(data['second_date'][data['second_date'] != 0.0].min() ) 

data['Days'] = (data['second_date'] - data['first_date'])
data['Days'].fillna(data['Days'].mean(), inplace = True)


data['id_ta'] = data['ID_TA'].apply(lambda x: int(x[1:]))
plt.rcParams['figure.figsize'] = (15,10)
sns.heatmap(data.drop(['sample'], axis=1).corr(),)

# data = data.drop(columns = ['Restaurant_id','Cuisine Style','first_date','second_date', 'Reviews', 'reviews', 'reviews_dates',  'URL_TA', 'ID_TA'], axis = 1)  # увеличивает значение MAE
data = data.drop(columns = ['Restaurant_id','Cuisine Style', 'Reviews', 'reviews', 'reviews_dates', 'URL_TA', 'ID_TA'], axis = 1)  
data = data.fillna(0)
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
y_pred = np.round(y_pred*2)/2
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)
