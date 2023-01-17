# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



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
df_train.head(5)
df_test.info()
df_test.head(5)
sample_submission.head(5)
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
df.info()
#Renaming the columns



df.columns = ['Restaurant_id', 'City', 'Cuisine_Style', 

              'Ranking', 'Price_Range','Number_of_Reviews', 

              'Reviews', 'URL_TA', 'ID_TA','sample', 'Rating']
df.sample(5)
df.Reviews[1]
df.info()
# Checking how many Price Ranges there are in the data set



df["Price_Range"].value_counts()
df['Price_Range'].isna().value_counts()
# map: '$' = 1, '$$ - $$$' = 2 ,'$$$$' = 3



def price_type(x):

    if x == '$':

        return 1

    elif x == '$$ - $$$':

        return 2

    elif x == '$$$$':

        return 3

    else:

        return 2

    

df["Price_Range"]= df["Price_Range"].apply(price_type)

#df["Price_Range"].value_counts()
# Add city population to the data frame. List of populations is compiled from the data from Wikipedia



city_pop = pd.read_csv('/kaggle/input/city-popscsv/city_pops.csv')

df = df.merge( city_pop, on='City', how = 'left') 



#flagging Capital sites in column Capital



city_capital = pd.read_csv('/kaggle/input/country-capital-listcsv/country_capital_list.csv')

city_capital = city_capital.capital.tolist() 



def capital(city):

    if city in city_capital:

        return 1

    else:

        return 0

df['Capital'] = df.City.apply(capital)



# City dummy variables using One-hot Encoding (get_dummies)



df['Cityi'] = df['City']

df = pd.get_dummies(df, columns=[ 'Cityi',], dummy_na=True)

#df[df.population.isna() == True].City.value_counts()

#df
# adding feature - markinng if number of reviews existed originally



df['Number_of_Reviews_Was_NAN'] = df['Number_of_Reviews'].isna()
#If number of reviews is NaN, we Assume there was no review for the restaurant 

df['Number_of_Reviews'].fillna(0, inplace=True)
# the below worked worse as an estimate, so we keep 0 instead:



# we will fill in NaNs for Number of Reviews with a median reviews for each price range in the city.



# Creating a look-up table for median reviews by city and price type

# rev_fill = df.groupby(['City', 'Price_Range']).Number_of_Reviews.median()

# rev_fill=rev_fill.reset_index()

# rev_fill.columns = ['City', 'Price_Range', 'med_reviews']

# rev_fill['citypricerange'] = rev_fill['City']+rev_fill['Price_Range'].astype(str)

# rev_med=rev_fill[['citypricerange', 'med_reviews']]

# rev_med= rev_med.round({'med_reviews':1})

#rev_med







# key for df and merging the data

# df['citypricerange'] = df['City']+df['Price_Range'].astype(str)

# df = df.merge(rev_med, on = 'citypricerange', how = 'left')







# # filling in the NaNs in the Number of Reviews with the calculated above medians

# def avg_rev(row):

#     if pd.isna(row['Number_of_Reviews']) == True:

#         return row['med_reviews']

#     else:

#         return row['Number_of_Reviews']

    

# dummy = df.apply(lambda row: avg_rev(row), axis = 1)



# df['Number_of_Reviews'] = dummy
# If cuisine type is missing, we assume that there was only one type of cuisine in this place

df['Cuisine_Style'] = df['Cuisine_Style'].fillna('')



# Clean up the Cuisine_Style column 

df['Cuisine_Style'] = df.Cuisine_Style.apply( lambda x: x[1:-1])

df['Cuisine_Style'] = df.Cuisine_Style.apply( lambda x: str(x).replace(' ', ''))

df['Cuisine_Style'] = df.Cuisine_Style.apply( lambda x: str(x).replace('  ', ''))

df['Cuisine_Style'] = df.Cuisine_Style.apply( lambda x: str(x).replace("\'", ''))



# We will now count the number of Cuisine Styles in each restaurant and add this as additional feature:



#count number of cuisines and record it into Number_Cuisines column

import re

df['Number_Cuisines'] = df['Cuisine_Style'].apply(lambda x : len(re.findall(",", x))+1)
#Another feature: type of Cuisine. For this we split Cuisine_Style column into dummy variables



xxx = df['Cuisine_Style'].str.split(',')

cousines_list = pd.get_dummies(xxx.apply(pd.Series).stack(), prefix='cus').sum(level=0)

cousines_list.rename(columns={'cus_':'cus_Unknown'}, inplace=True)

df = pd.merge(df, cousines_list, left_index=True, right_index=True, how='left')
#lets flag restaurants that have the most common cuisine style for their City into column "Popular_cuisine"



#first we create a lookup table for the most frequent cuisine in each city:



new_df = df[['City', 'Cuisine_Style']]

new_df = new_df[new_df.Cuisine_Style != '']



s = new_df['Cuisine_Style'].str.split(',').apply(pd.Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'Cuisine_Style'

del new_df['Cuisine_Style']

new_df = new_df.join(s)

ss = pd.DataFrame(new_df.groupby(['City', 'Cuisine_Style'])['Cuisine_Style'].count()).stack().reset_index()

del ss['level_2']

ss.columns = ['City', 'Popular_cuisine', 'Cus_Freq']

popular = ss.sort_values('Cus_Freq', ascending=False).drop_duplicates('City')

popular = popular [['City', 'Popular_cuisine']]



#and merge it to the dataframe

df = pd.merge(df, popular, on ='City')



#Now let's flag whether the restaurant serves popular cuisine for this region:

def pop_cus(row):

    if row['Popular_cuisine'] in row['Cuisine_Style']: 

        return 1

    else:

        return 0

    

dum=df.apply(lambda row: pop_cus(row), axis = 1)



df["Popular_cuisine"]= dum
#And finally, lets drop columns not needed for the model

df = df.drop(columns=['City', 'Restaurant_id', 'Cuisine_Style','Reviews','URL_TA','ID_TA'])
df.info(verbose = True, null_counts = True)
# Теперь выделим тестовую часть

train_data = df.query('sample == 1').drop(['sample'], axis=1)

test_data = df.query('sample == 0').drop(['sample'], axis=1)



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