import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from datetime import datetime

import ast



%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



# Загружаем инструменты для категориальных признаков

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# всегда фиксируйте RANDOM_SEED и дату, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42



CURRENT_DATE = '03/15/2020'
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
data.Reviews[1]
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data.nunique(dropna=False)
data['Price Range'].value_counts()
def price_range(cell):

    if pd.isna(cell):

        return 2

    elif cell == '$':

        return 1

    elif cell == '$$ - $$$':

        return 2

    elif cell == '$$$$':

        return 3



data['Price Range'] = data['Price Range'].apply(price_range)

# data = pd.get_dummies(data, columns=['Price Range', ])
def make_list(list_string):

    list_string = list_string.replace('nan]', "'This is Nan']")

    list_string = list_string.replace('[nan', "['This is Nan'")

    result_list = ast.literal_eval(list_string)

    return result_list



data['Reviews'] = data['Reviews'].fillna("[[], []]")

data['Reviews'] = data['Reviews'].apply(make_list)
def delta_date(row):

    if len(row[1]) == 0 or len(row[1]) == 1:

        return 0

    

    elif len(row[1]) == 2:

        date1 = datetime.strptime(row[1][0],'%m/%d/%Y')

        date2 = datetime.strptime(row[1][1],'%m/%d/%Y')

        return abs(date1 - date2).days



data['Days between reviews'] = data['Reviews'].apply(delta_date)
current_date_dt = datetime.strptime(CURRENT_DATE,'%m/%d/%Y')



def since_last_days(row):

    if len(row[1]) == 0:

        date = datetime.strptime('01/01/2000','%m/%d/%Y')

    

    elif len(row[1]) == 1:

        date = datetime.strptime(row[1][0],'%m/%d/%Y')

    

    else:

        date1 = datetime.strptime(row[1][0],'%m/%d/%Y')

        date2 = datetime.strptime(row[1][1],'%m/%d/%Y')

        date = max(date1, date2)

    

    return (current_date_dt - date).days



data['Days since last review'] = data['Reviews'].apply(since_last_days)
data['Reviews count'] = data['Reviews'].apply(lambda x: len(x[0]))
# https://www.geeksforgeeks.org/python-nlp-analysis-of-restaurant-reviews/

# import re  

  

# # Natural Language Tool Kit 

# import nltk  

  

# nltk.download('stopwords')

  

# # to remove stopword 

# from nltk.corpus import stopwords 

  

# # for Stemming propose  

# from nltk.stem.porter import PorterStemmer 



# from sklearn.feature_extraction.text import CountVectorizer

# from sklearn.ensemble import RandomForestClassifier 

# from sklearn.model_selection import train_test_split

# from sklearn.metrics import confusion_matrix



# yelp = pd.read_csv('/kaggle/input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/yelp_labelled.txt',

#                    sep='\t', header=None, usecols=[0,1])

# yelp.columns = ['Reviews', 'Rating']
# def clean_corpus(my_list):

    

#     corpus = []

    

#     for item in my_list:  

      

#         # column : "Review", row ith 

#         review = re.sub('[^a-zA-Z]', ' ', item)  



#         # convert all cases to lower cases 

#         review = review.lower()  



#         # split to array(default delimiter is " ") 

#         review = review.split()  



#         # creating PorterStemmer object to 

#         # take main stem of each word 

#         ps = PorterStemmer()  



#         # loop for stemming each word 

#         # in string array at ith row     

#         review = [ps.stem(word) for word in review 

#                     if not word in set(stopwords.words('english'))]  



#         # rejoin all string array elements 

#         # to create back into a string 

#         review = ' '.join(review)   



#         # append each string to create 

#         # array of clean text  

#         corpus.append(review)

        

#     return corpus



# corpus = clean_corpus(yelp['Reviews'].to_list())



 

# # To extract max 1500 feature. 

# # "max_features" is attribute to 

# # experiment with to get better results 

# cv = CountVectorizer(max_features = 1500)  

  

# # X contains corpus (dependent variable) 

# X = cv.fit_transform(corpus).toarray()  

  

# # y contains answers if review 

# # is positive or negative 

# y = yelp.iloc[:, 1].values



# # experiment with "test_size" 

# # to get better results 

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

  

# # n_estimators can be said as number of 

# # trees, experiment with n_estimators

# # to get better results  

# model = RandomForestClassifier(n_estimators = 501, criterion = 'entropy', random_state = 42) 

                              

# model.fit(X_train, y_train) 
# y_pred = model.predict(X_test)

  

# cm = confusion_matrix(y_test, y_pred) 



# cm
# comment1 = []

# comment2 = []



# for review in data['Reviews']:

#     if len(review[0]) == 2:

#         comment1.append(review[0][0])

#         comment2.append(review[0][1])

#     elif len(review[0]) == 1:

#         comment1.append(review[0][0])

#         comment2.append('Good')

#     else:

#         comment1.append('Good')

#         comment2.append('Good')
# corpus1 = clean_corpus(comment1)



# cv = CountVectorizer(max_features = 1500)  

  

# # X contains corpus (dependent variable) 

# corpus_X1 = cv.fit_transform(corpus1).toarray()

# y_pred1 = model.predict(corpus_X1)



# corpus2 = clean_corpus(comment2)

# cv = CountVectorizer(max_features = 1500)  

  

# # X contains corpus (dependent variable) 

# corpus_X2 = cv.fit_transform(corpus2).toarray()

# y_pred2 = model.predict(corpus_X2)
# data['Comment1'] = y_pred1

# # data['Comment2'] = y_pred2
# data['Comment average'] = (data['Comment1'] + data['Comment2']) / 2



# data = data.drop(['Comment1', 'Comment2'], axis=1)
from collections import Counter



def get_most_common(row, count):

    new = []

    c = Counter(row)

    most = c.most_common(count)



    for item in most:

        new.append(item[0])

    new_str = ','.join(new)

    return new



data['Cuisine Style'] = data['Cuisine Style'].str[2:-2].str.split("', '")

cuisines_list = data['Cuisine Style']

city_and_cuisines = pd.concat([data['City'], cuisines_list], axis=1)

city_and_cuisines = city_and_cuisines.dropna()

cities_grouped = city_and_cuisines.groupby('City').agg({'Cuisine Style': sum}).reset_index()

cities_grouped.columns = ['City', 'Cuisine List']

cities_grouped_mean = city_and_cuisines.groupby('City')['Cuisine Style'].apply(lambda x: round(np.mean(x.str.len()))).reset_index()

cities_grouped_mean.columns = ['City2', 'Cuisine Count']

new = pd.concat([cities_grouped, cities_grouped_mean], axis=1, join='inner')

new = new.drop('City2', axis=1)

new['Common Cuisine'] = new.apply(lambda x: get_most_common(x['Cuisine List'], x['Cuisine Count']), axis=1)

new = new.set_index('City')

common_cuisine_dict = new['Common Cuisine'].to_dict()

data['Cuisine Style'] = data['Cuisine Style'].fillna(data['City'].map(common_cuisine_dict))
df_new = data['Cuisine Style']

df_new_dummy = df_new.apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0, downcast='infer')



data = pd.merge(data, df_new_dummy, left_index=True, right_index=True, how='left')
data['Cuisine count'] = data['Cuisine Style'].str.len()
# Add capital sity 0 or 1, 'Edinburgh' is not capital



capitals = ['Paris', 'Stockholm', 'London', 'Berlin',

       'Bratislava', 'Vienna', 'Rome', 'Madrid',

       'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen',

       'Amsterdam', 'Lisbon', 'Prague', 'Oslo',

       'Helsinki', 'Ljubljana', 'Athens', 'Luxembourg',]





def is_capital(city):

    if city in capitals:

        return 1

    else:

        return 0



data['is_capital'] = data['City'].apply(is_capital)
cities = {

    'London': 8567000, 

    'Paris': 9904000, 

    'Madrid': 5567000,

    'Barcelona': 4920000,

    'Berlin': 3406000,

    'Milan': 2945000,

    'Rome': 3339000,

    'Prague': 1162000,

    'Lisbon': 2812000,

    'Vienna': 2400000,

    'Amsterdam': 1031000,

    'Brussels': 1743000,

    'Hamburg': 1757000,

    'Munich': 1275000,

    'Lyon': 1423000,

    'Stockholm': 1264000,

    'Budapest': 1679000,

    'Warsaw': 1707000,

    'Dublin': 1059000,

    'Copenhagen': 1085000,

    'Athens': 3242000,

    'Edinburgh': 504966,

    'Zurich': 1108000,

    'Oporto': 1337000,

    'Geneva': 1240000,

    'Krakow': 756000,

    'Oslo': 835000,

    'Helsinki': 1115000,

    'Bratislava': 423737,

    'Luxembourg': 107260,

    'Ljubljana': 314807,

}



data['Population'] = data['City'].map(cities)
rest_count = data.groupby('City')['Restaurant_id'].count().to_dict()

data['Total count of restaurants'] = data['City'].map(rest_count)

data['Relative ranking'] = data['Ranking'] / data['Total count of restaurants']



# data.drop(['Ranking', 'rest_total_count'], axis = 1, inplace=True)
data['People per restaurant'] = data['Population'] / data['Total count of restaurants']
countries = {

    'London': 'GB',

    'Paris': 'FR',

    'Madrid': 'ES',

    'Barcelona': 'ES',

    'Berlin': 'DE',

    'Milan': 'IT',

    'Rome': 'IT',

    'Prague': 'CZ',

    'Lisbon': 'PT',

    'Vienna': 'AT',

    'Amsterdam': 'NL',

    'Brussels': 'BE',

    'Hamburg': 'DE',

    'Munich': 'DE',

    'Lyon': 'FR',

    'Stockholm': 'SE',

    'Budapest': 'HU',

    'Warsaw': 'PL',

    'Dublin': 'IE',

    'Copenhagen': 'DK',

    'Athens': 'GR',

    'Edinburgh': 'GB',

    'Zurich': 'CH',

    'Oporto': 'PT',

    'Geneva': 'CH',

    'Krakow': 'PL',

    'Oslo': 'NO',

    'Helsinki': 'FI',

    'Bratislava': 'SK',

    'Luxembourg': 'LU',

    'Ljubljana': 'SI',

}



data['Country'] = data['City'].map(countries)



countries_le = LabelEncoder()

countries_le.fit(data['Country'])

data['Country Code'] = countries_le.transform(data['Country'])
data
data['Reviews on people'] = data['Reviews count'] / data['People per restaurant']
cities_le = LabelEncoder()

cities_le.fit(data['City'])

data['City Code'] = cities_le.transform(data['City'])
restaurant_chain = set()

for chain in data['Restaurant_id']:

    restaurant_chain.update(chain)

    

def find_item1(cell):

    if item in cell:

        return 1

    return 0



for item in restaurant_chain:

    data['Restaurant chain'] = data['Restaurant_id'].apply(find_item1)
data['ID_TA code'] = data['ID_TA'].apply(lambda x: int(x[1:]))
df_train['Ranking'].describe()
object_columns = [s for s in data.columns if data[s].dtypes == 'object']

data.drop(object_columns, axis = 1, inplace=True)



df_preproc = data

df_preproc.sample(10)
df_preproc.info(verbose=True)
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

# y_pred = model.predict(X_test)
# Округляем результаты с точностью 0.5



def rating_round(x, base=0.5):

    return base * round(x/base)



def predict(ds):

    return np.array([rating_round(x) for x in model.predict(ds)])



y_pred = predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# Best MAE: 0.165875
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
# predict_submission = model.predict(test_data)
predict_submission = predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)