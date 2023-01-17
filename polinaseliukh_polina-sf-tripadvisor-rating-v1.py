# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline


from sklearn.preprocessing import MultiLabelBinarizer
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
df_train.head(5)
df_test.info()
df_test.head(5)
sample_submission.head(5)
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать,
                      # по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
# Number of Reviews
data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
data.nunique(dropna=False)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
# data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
# data['Price Range'].fillna(data['Price Range'].mode()[0], inplace=True)
price_range_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3, None : 0}
data['Price_Range_num'] = data['Price Range'].replace(to_replace=price_range_dict)

sns.countplot(x='Rating', hue='Price_Range_num', data=data)
data[data['Reviews'].isna()]
print(data['Reviews'].nunique()) 
# уникальных отзывов 41857
print(data['Reviews'].value_counts())
# из них 8112 - пустых
new_df = pd.DataFrame(data['Reviews'].value_counts().values)
new_df[0].value_counts()
# 30  отзывов встречается по 2 раза
# 41826 - истинно уникальных

data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: None if pd.isna(x)
                                                else x.strip("[]"))
data['Cuisine Style'].fillna('NO INFO', inplace=True)

# узнать какая кухня встречается чаще всего
from collections import Counter
import re
cuis_st_counter = Counter()
for i in data['Cuisine Style']:
    l = re.sub('\s\'|\'', '', i).split(',')
    cuis_st_counter.update(l)
cuis_st_counter.most_common()
'''вспомогательная функция из строки сделать список'''
def str_to_list(string):
        _list = ["[","]","'"]

        if string != None:
            for i in _list:
                string = str(string).replace(i,'')
            return string.split(', ')
        return string

    
'''Возвращает измененный DataFrame
Признак Cuisine Style модифицируется в 3 dummy-столбца с самыми популярными кухнями 
(Vegetarian Friendly, European, Mediterranean)'''
def get_top_cuisine_style_dummies(dataF):
    dataF['Cuisine Style'] = dataF['Cuisine Style'].apply(str_to_list)
    # применяем MultiLabelBinarizer, он делает то, что мы и хотим.
    mlb = MultiLabelBinarizer()
    dataF = dataF.join(pd.DataFrame(mlb.fit_transform(dataF.pop('Cuisine Style')), 
                                    index=dataF.index, columns=mlb.classes_))
    columns_to_drop = mlb.classes_.tolist()
    columns_to_drop.remove('Vegetarian Friendly')
    columns_to_drop.remove('European')
    columns_to_drop.remove('Mediterranean')
    dataF.drop(columns_to_drop, axis=1, inplace=True)
    return dataF
data = get_top_cuisine_style_dummies(data)
data.info()


data[data['URL_TA']=='/Restaurant_Review-g187514-d7342803-Reviews-Los_Hierros-Madrid.html']
# display(data[data.duplicated(['URL_TA', 'ID_TA'], keep=False)].sort_values(by='URL_TA'))
# 74 записи (37 пар) где совпадают URL_TA, ID_TA
print(len(data[data.duplicated(['URL_TA', 'ID_TA'], keep=False)]
          .sort_values(by='URL_TA').query('sample==1')))
print(len(data[data.duplicated(['URL_TA', 'ID_TA'], keep=False)]
          .sort_values(by='URL_TA').query('sample==0')))


data['Restaurant_id'].value_counts()[0:20]
data[data['Restaurant_id']=='id_206']

plt.rcParams['figure.figsize'] = (10,7)
df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['Ranking'][df_train['City'] == x].hist(bins=100)
plt.show()
'''подсчет абсолютного значения позиции ресторана в списке ресторанов (Ranking)
Для каждого города найти максимальное и минимальное значение Ranking
Выполнить minmax-пересчет признака Ranking
Ranking_Absolute := (x - x_min) / (x_max - x_min)
Значение 0 - соответстует лучшему ресторану, 1 - ресторану, замыкающему рейтинговый список'''

def get_ranking_absolute(dataF):
    row_list = []
    for x in (dataF['City'].value_counts()).index:
        dict_row = {}
        dict_row.update({'City': x, 
                         'max_rank_in_city': dataF['Ranking'][dataF['City'] == x].max(),
                         'min_rank_in_city': dataF['Ranking'][dataF['City'] == x].min()})
#                          'min_rank_in_city': 1})

        row_list.append(dict_row)

    df_city_min_max_ranking = pd.DataFrame(row_list)
    join_df = dataF.merge(df_city_min_max_ranking, how='left', on='City')
    
    rez = (join_df['Ranking']-join_df['min_rank_in_city']) / \
        (join_df['max_rank_in_city']-join_df['min_rank_in_city'])
    return rez
df_train['Ranking_Absolute'] = get_ranking_absolute(df_train)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit_transform()
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['Ranking_Absolute'][df_train['City'] == x].hist(bins=100)
plt.show()
df_train['Ranking_Absolute'].hist(bins=100)
sns.jointplot(x='Rating', y='Ranking_Absolute', data=df_train)
'''возвращает DataFrame (City, max_rank_in_city) 
max_rank_in_city соответствует максимальному значению признака Ranking для города
а именно какое максимальное место в списке ресторанов города встречалось - 
тракутем как Сколько всего ретсоранов рассматривается в городе для существующего сета данных'''

def get_max_rank_in_city(dataF):
    row_list = []
    for x in (dataF['City'].value_counts()).index:
        dict_row = {}
        dict_row.update({'City': x, 
                         'max_rank_in_city': dataF['Ranking'][dataF['City'] == x].max()})
        row_list.append(dict_row)

    df_city_max_rank = pd.DataFrame(row_list)
    return df_city_max_rank



'''Подсчет относительного количества отзывов ресторана в городе
функция возвращает значение = количество отзывов у ресторана / суммарное количество отзывов в городе'''

def get_Number_of_reviews_norm(dataF):
    reviews_in_city = dataF.groupby(by=['City'])['Number of Reviews'].sum()
    dataF = dataF.merge(reviews_in_city, on='City', how='left', suffixes=[None, '_in_city'])
    rez = dataF['Number of Reviews'] / dataF['Number of Reviews_in_city']
    return rez
    


df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
plt.rcParams['figure.figsize'] = (15,10)
sns.heatmap(data.drop(['sample'], axis=1).corr(),annot=True, cmap='YlGn')

# на всякий случай, заново подгружаем данные
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0  # помечаем где у нас тест
df_test['Rating'] = 0  # в тесте у нас нет значения Rating, мы его должны предсказать, 
                       # по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
    # ################### 1. Предобработка ############################################################## 
    # убираем не нужные для модели признаки
    df_output.drop(['Restaurant_id','ID_TA','URL_TA'], axis = 1, inplace=True)
    
    
    # ################### 2. NAN ############################################################## 
    # Далее заполняем пропуски
    
    # [+] Number_of_Reviews_isNAN отметим пропуски
    # не улучшает модель, не являетс очень значимым признаком. не добавляем
#     df_output['Number_of_Reviews_isNAN'] = pd.isna(df_output['Number of Reviews']).astype('uint8')
    
    df_output['Number of Reviews'].fillna(0, inplace=True)
    
    # изменение данных Cuisine Style: remove symbols '[' ']'
    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(lambda x: None if pd.isna(x)
                                                else x.strip("[]"))
    df_output['Cuisine Style'].fillna('NO INFO', inplace=True)

    
    # Price_Range_isNAN отметим присутствует ли значение
#     df_output['Price_Range_isNAN'] = pd.isna(df_output['Price Range']).astype('uint8')
    # заполнение пропусков Price Range самым часто встречаемым значением
    # df['Price Range'].mode() возвращает список
#     df_output['Price Range'].fillna(df_output['Price Range'].mode()[0], inplace=True)

    df_output['Reviews_isNAN'] = pd.isna(df_output['Reviews']).astype('uint8')
    df_output['Reviews'].fillna('[[], []]', inplace=True)
    
    
    
    
    
    # ################### 4. Feature Engineering ####################################################
    # 
    # [+] - обозначение столбцов, которые будут анализироваться на предмет "добавить в модель"
    # (другие столбцы добавляются как вспомогательные, в модель не войдут)
    
    # [+] добавим новый столбец Cuisine_cnt - количество типов кухонь в ресторане
    # Если тип кухни 'NO INFO' - считаем, что предлагается 1 тип
    df_output['Cuisine_cnt'] = df_output['Cuisine Style'].apply(lambda x: 1 if x == 'NO INFO'
                                              else len(x.split(', ')))
    
#     df_output = get_top_cuisine_style_dummies(df_output)

    # [+] Price_Range_num числовое значение (ординарный признак) цен в ресторане
    price_range_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3, None: 0}
    df_output['Price_Range_num'] = df_output['Price Range'].replace(to_replace=price_range_dict)

#     price_range_dict2 = {'$': 1, '$$ - $$$': 2, '$$$$': 3, None: 0}
#     df_output['Price_Range_forDummie'] = df_output['Price Range'].replace(to_replace=price_range_dict)

    
    # блок про даты отзывов
    # review_dates содержит список дат, когда оставили отзывы (2шт), которые отображаются на странице
    df_output['review_dates'] = df_output['Reviews'].apply(lambda x: x.split(
        "], [")[1].strip('][').replace("'", '').split(', '))

    # дата 1-го отзыва
    df_output['review_1st_date'] = pd.to_datetime(
        df_output['review_dates'].apply(lambda x: x[0]))
    
    # дата 2-го отзыва
    df_output['review_2nd_date'] = pd.to_datetime(df_output['review_dates'].apply(lambda x: x[1] if len(x) == 2
                                                                else ''))
    
    # [+] Days_form_last_review сколько дней прошло с последнего отзыва
    df_output['Days_from_last_review'] = (pd.to_datetime(datetime.now()) -
                               pd.to_datetime(df_output['review_dates'].apply(lambda x: max(x)))).apply(lambda x: x.days)
    df_output['Days_from_last_review'].fillna(0, inplace=True)

    # [+] year_of_last_review год самого свежего отзыва
    df_output['year_of_last_review'] = pd.to_datetime(df_output['review_dates'].apply(lambda x: max(x))).apply(lambda x: x.year)
    df_output['year_of_last_review'].fillna(0, inplace=True)

    # [+] Review_date_delta -  разница в днях между 1ой и 2ой датами отзывов
    # (? на сколько отзывы актуальны относительно друг друга ? )
    review_delta = np.abs(df_output['review_1st_date']-df_output['review_2nd_date'])
    df_output['Review_date_delta'] = review_delta.apply(lambda x: x.days)
    
    df_output['review_1st_date'].fillna(0, inplace=True)
    df_output['review_2nd_date'].fillna(0, inplace=True)
    df_output['Review_date_delta'].fillna(0, inplace=True)
    
    # [+] Ranking_Absolute - абсолютное значение позиции ресторана в списке ресторанов своего города.
    # Значение 0 - соответстует лучшему ресторану, 1 - ресторану, замыкающему рейтинговый список
    df_output['Ranking_Absolute'] = get_ranking_absolute(df_output)

    # [+] Number of Reviews_norm - относительно количество отзывов в ресторане 
    # (относительно общего количества отзывов в городе)
    # - какая часть всех отзывов в городе приходится на отзывы про конкретный ресторан
    df_output['Number of Reviews_norm'] = get_Number_of_reviews_norm(df_output)
    
    
    # ################### 3. Encoding ############################################################## 
    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
    
    df_output = pd.get_dummies(df_output, columns=['City'], dummy_na=True)
    
#     df_output = pd.get_dummies(df_output, columns=['Price_Range_num'])
    
    
    # ################### 5. Clean #################################################### 
    # убираем признаки которые не нужны 
    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим
    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']
    df_output.drop(object_columns, axis = 1, inplace=True)
    
    
    # убираем признаки, которые могут привести к переобучению модели или мало на нее влияют
#     df_output.drop('Ranking', axis=1, inplace=True)
#     df_output.drop('year_of_last_review', axis=1, inplace=True)

    
    
    return df_output
df_preproc = preproc_data(data)
df_preproc.sample(10)
sns.heatmap(df_preproc.drop(['sample'], axis=1).corr(), cmap='YlGn')
df_preproc.info()
# Теперь выделим тестовую часть
train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

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
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)
