import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')

df_train.columns = ['restaurant_id','city','cuisine_style','ranking','rating','price_range','reviews_number','reviews',

             'url_ta','id_ta']

df_test.columns = ['restaurant_id','city','cuisine_style','ranking','price_range','reviews_number','reviews',

             'url_ta','id_ta']

df_train['sample'] = 1 

df_test['sample'] = 0 

df_test['rating'] = 0 

df = df_test.append(df_train, sort=False).reset_index(drop=True)

df.info()
RANDOM_SEED = 42

!pip freeze > requirements.txt
df['number_of_reviews_is_nan'] = pd.isna(df['reviews_number']).astype('uint8')

df['price_range_is_nan'] = pd.isna(df['price_range']).astype('uint8')
rest_count = df.city.value_counts()

def get_rest_count(value):

    return rest_count[value]

#чем больше к единице, тем выше ранг ресторана

df['relative_ranking'] = 1-(df['ranking'] / df['city'].map(df.groupby(['city'])['ranking'].max()))
df['price_range'].value_counts()

replace_price_range = {'$$ - $$$':1,'$':0,'$$$$':2,float('nan'):1}

df['price_range'] = df['price_range'].map(replace_price_range)

df['mean_price'] = df['city'].map(df.groupby('city')['price_range'].mean())
capitals = ['London','Paris','Madrid','Berlin','Rome','Prague','Lisbon','Vienna','Amsterdam',

            'Brussels','Stockholm','Budapest','Warsaw','Copenhagen','Athens','Dublin','Oslo','Helsinki',

            'Bratislava','Luxembourg','Ljubljana','Edinburgh']

def check_if_capital(value):

    return 1 if value in capitals else 0

df['is_capital'] = df['city'].apply(check_if_capital)
cuisine_dict = {}

cuisine_list = []

import re

re_cuisine = re.compile(r"'([^']+)'")

def get_cuisines_list(value):

    if type(value) is not str:

        cuisine_list.append('Vegetarian Friendly')

        return ['Vegetarian Friendly']#как медианное значение

    else:

        cuisine_list.extend(re_cuisine.findall(value))

        return re_cuisine.findall(value)

def create_cuisine_dictionary(raw):

    if raw[1] not in cuisine_dict.keys():

        cuisine_dict[raw[1]] = []

    cuisine_dict[raw[1]].extend(raw[2])



df['cuisine_style'] = df['cuisine_style'].apply(get_cuisines_list)

df.apply(create_cuisine_dictionary,axis=1)
for key in cuisine_dict.keys():

    cuisine_series = pd.Series(cuisine_dict[key]).value_counts()

    cuisines_count = cuisine_series.sum()

    cuisines = {}

    for cuisine in cuisine_series.index:

        cuisines[cuisine] = cuisine_series[cuisine] / cuisines_count

    cuisine_dict[key] = cuisines
df['cuisine_count'] = df['cuisine_style'].apply(lambda x: len(x) if type(x) is list else 1)
df['cuisine_count_mean'] = df['city'].map(df.groupby('city')['cuisine_count'].mean())
def get_cuisines_popularity(raw):

    result = 0

    if type(raw[2]) is list:

        for item in raw[2]:

            result = result+cuisine_dict[raw[1]][item]

    return result

df['cuisine_popularity'] = df.apply(get_cuisines_popularity,axis=1)
df['reviews_number'].fillna(0,inplace=True)
import datetime

pattern = re.compile('\d+\W\d+\W\d+')

def choose_date(reviews):

    dates = []

    if type(reviews) is str:

        dates = pattern.findall(reviews)

    return dates

def compute_days_reviews(value):

    if type(value) is list:

        if len(value) == 2:

            return abs((pd.to_datetime(str(value[0]))-pd.to_datetime(str(value[1]))).days)

    return 0



df['dates'] = df['reviews'].apply(choose_date)

df['days_between_reviews'] = df['dates'].apply(compute_days_reviews)
price_dummies = pd.get_dummies(df['price_range'])

df = pd.concat([df, price_dummies], axis = 1)
cuisines = pd.Series(cuisine_list).unique()

for cuisine in cuisines:

    df[cuisine] = df['cuisine_style'].apply(lambda x: 1 if cuisine in x else 0)
city_dummies = pd.get_dummies(df['city'])

df = pd.concat([df, city_dummies], axis = 1)
#данные взяты из Википедии

average_income = {'Paris': 3332, 'Stockholm': 2893, 'London': 2703, 'Berlin': 4392, 'Munich': 4392, 'Oporto': 1288,

       'Milan':2726, 'Bratislava':1283, 'Vienna':2940, 'Rome':2726, 'Barcelona':2133, 'Madrid':2133,

       'Dublin':3671, 'Brussels':3930, 'Zurich':6244, 'Warsaw':1253, 'Budapest':1187, 'Copenhagen':6192,

       'Amsterdam':3238, 'Lyon':3332, 'Hamburg':4392, 'Lisbon':1288, 'Prague':1454, 'Oslo':5450,

       'Helsinki':3908, 'Edinburgh':2703, 'Geneva':6244, 'Ljubljana':1914, 'Athens':1203,

       'Luxembourg':5854, 'Krakow':1253}

df['average_income'] = df['city'].map(average_income)
#данные взяты из Википедии

population_density = {'Paris': 20781, 'Stockholm': 5139.7, 'London': 5667, 'Berlin': 4088, 'Munich': 4713, 'Oporto': 5703,

       'Milan':7588.97, 'Bratislava':1171, 'Vienna':4502.88, 'Rome':2234, 'Barcelona':15779, 'Madrid':8653.5,

       'Dublin':3689, 'Brussels':5497, 'Zurich':4666, 'Warsaw':3449, 'Budapest':3330.5, 'Copenhagen':6214.7,

       'Amsterdam':4768, 'Lyon':10041, 'Hamburg':2388.57, 'Lisbon':6243.9, 'Prague':2506, 'Oslo':1483.41,

       'Helsinki':899, 'Edinburgh':4140, 'Geneva':12589, 'Ljubljana':1736, 'Athens':7500,

       'Luxembourg':2240, 'Krakow':2344}



df['population_density'] = df['city'].map(population_density)
df = df.select_dtypes(include = ['float64', 'int64'])

df = df.drop(['ranking','price_range'], axis = 1)
train_data = df.query('sample == 1').drop(['sample'], axis=1)

test_data = df.query('sample == 0').drop(['sample'], axis=1)



y = train_data.rating.values# наш таргет

X = train_data.drop(['rating'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics



def result_round(value):

    return np.fromiter(map(lambda x: round(x*2,0)/2,value), dtype=np.float)



model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)



model.fit(X_train, y_train)



y_pred = model.predict(X_test)



y_pred_round = result_round(y_pred)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred_round))