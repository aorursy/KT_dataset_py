# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
from datetime import datetime

from datetime import timedelta

import datetime

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

import re

from sklearn.model_selection import train_test_split # Загружаем специальный инструмент для разбивки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
pd.set_option('display.max_columns', 200)

DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')

display(df_train.head(2))

display(df_test.head(2))
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['Sample'] = 1 # помечаем где у нас трейн

df_test['Sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем



df.sample(5)
df.columns = ['Restaurant_id', 'City', 'Cuisine_Style', 'Ranking', 'Price_Range',

       'Number_of_Reviews', 'Reviews', 'URL_TA', 'ID_TA', 'Sample', 'Rating']
df.info()
df['Restaurant_id'] = df['Restaurant_id'].apply(lambda x: float(x[3:]))

df.head(3)
print(f'Количество городов в датасете - {len(df.City.unique())}')

print(df.City.unique())
not_capital = ["Krakow", "Lyon","Zurich","Hamburg","Barcelona","Oporto","Munich","Milan","Geneva"]
capital_rest=[]

other_city_rest=[]

for i in df['City']:

    if i in not_capital:

        other_city_rest.append(i)

    else:

        capital_rest.append(i)

        

print(f'Количество ресторанов в столицах:{len(capital_rest)}')

print(f'Количество остальных ресторанов :{len(other_city_rest)}')
df['Capital_restaurant']=df['City'].apply(lambda x: 0 if x in not_capital else 1)

df[['Capital_restaurant']].sample(5)
df_City_dummies = pd.get_dummies(df['City'], dummy_na=False).astype('float64')

df = pd.concat([df,df_City_dummies], axis=1)
display((pd.isna(df['Cuisine_Style']).astype('float64')).value_counts())
# проведем обработку значений переменной

df['Cuisine_Style']= df['Cuisine_Style'].str.replace(r'[\[\]\']','')

# заполним пропуски значением 'Unspecified'

df['Cuisine_Style'] = df['Cuisine_Style'].fillna('Unspecified')

#посчитаем количество кухонь в каждом ресторане

df['Сount_Сuisine'] = df.Cuisine_Style.apply(lambda x: len(x.split(',')))
df.Ranking.describe()
df[['Restaurant_id','Ranking']].corr()
df = df.drop(['Restaurant_id'], axis=1)
Сity_population= {'London' : 8539, 'Paris' : 2197, 'Madrid' : 3222, 'Barcelona' : 1621, 

                        'Berlin' : 3723, 'Milan' : 1342, 'Rome' : 2869, 'Prague' : 1281, 

                        'Lisbon' : 506, 'Vienna' : 1889, 'Amsterdam' : 866, 'Brussels' : 179, 

                        'Hamburg' : 1718, 'Munich' : 1450, 'Lyon' : 516, 'Stockholm' : 961, 

                        'Budapest' : 1745, 'Warsaw' : 1790, 'Dublin' : 554, 

                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 

                        'Zurich' : 402, 'Oporto' : 249, 'Geneva' : 184, 'Krakow' : 755, 

                        'Oslo' : 693, 'Helsinki' : 643, 'Bratislava' : 426, 

                        'Luxembourg' : 120, 'Ljubljana' : 284}

df['Сity_population'] = df.apply(lambda row: Сity_population[row['City']], axis = 1)
mean_Ranking_on_City = df.groupby(['City'])['Ranking'].mean()

df['mean_Ranking_on_City'] = df.City.map(mean_Ranking_on_City)

df['norm_Ranking_on_Popul_in_City'] = (df['Ranking'] - df['mean_Ranking_on_City']) / df['Сity_population']
display(pd.isna(df['Price_Range']).astype('float64').value_counts())
df['Price_Range'].value_counts()
df.loc[df['Price_Range'] == '$$$$', ['Price_Range']] = 3

df.loc[df['Price_Range'] == '$$ - $$$', ['Price_Range']] = 2

df.loc[df['Price_Range'] == '$', ['Price_Range']] = 1

df['Price_Range'] = df['Price_Range'].fillna(2)

df['Price_Range'].value_counts()
display(pd.isna(df['Number_of_Reviews']).astype('float64').value_counts())
means1 = df.groupby('City')['Number_of_Reviews'].mean()

df['Number_of_Reviews_means'] = round(df.City.map(means1),0)
df['Number_of_Reviews'] = df['Number_of_Reviews'].fillna(df['Number_of_Reviews'].mean())

df['Number_of_Reviews'].value_counts()
df['Reviews'] = df['Reviews'].fillna('[[], []]')

df['Reviews'] = df['Reviews'].str.replace(r'\[\[\], \[\]\]','None')

s = df['Reviews'].str.split(r"'\], \['",expand=True)

s.columns = ['comments','dates']

s1 = s['comments'].str.split(r"', '",expand=True)

s1.columns = ['comment1','comment2']

s2 = s['dates'].str.split(r"', '",expand=True)

s2.columns = ['date1','date2']

s3 = pd.concat([s1,s2],axis=1)

s3.date1 = s3.date1.str.replace(r'[\[\]\']','')

s3.date2 = s3.date2.str.replace(r'[\[\]\']','')

df_new = pd.concat([df,s3],axis=1)
date1null = pd.isna(df_new['date1']).astype('float64')

date2null = pd.isna(df_new['date2']).astype('float64')

df_new['len_comment'] = (1 - date1null) + (1 -date2null)

df_new[['len_comment']].head()
df_new['date1'] = pd.to_datetime(df_new['date1'])

df_new['date2'] = pd.to_datetime(df_new['date2'])

df_new['diff_date']=abs(df_new['date1']-df_new['date2'])

df_new.diff_date = df_new.diff_date.dt.days

df_new.diff_date = df_new.diff_date.fillna(0)
dayspass = df_new['date1'].max()-df_new['date1']

dayspass2 = df_new['date1'].max()-df_new['date2']

df_new['dayspass'] = np.where(dayspass >= dayspass2, dayspass, dayspass2)

df_new['dayspass'] = df_new['dayspass'].fillna(pd.to_timedelta(0))

df_new['dayspass'] = df_new['dayspass'].dt.days
df_new.drop(['Reviews','comment1','comment2','date1','date2'], axis='columns', inplace=True)
df_new.info()
df_new.drop(df_new.select_dtypes(['object']), inplace=True, axis=1)

df_new.head()
df_new.columns
df_new2 = df_new.drop(['Amsterdam', 'Athens', 'Barcelona', 'Berlin', 'Bratislava', 'Brussels',

       'Budapest', 'Copenhagen', 'Dublin', 'Edinburgh', 'Geneva', 'Hamburg',

       'Helsinki', 'Krakow', 'Lisbon', 'Ljubljana', 'London', 'Luxembourg',

       'Lyon', 'Madrid', 'Milan', 'Munich', 'Oporto', 'Oslo', 'Paris',

       'Prague', 'Rome', 'Stockholm', 'Vienna', 'Warsaw', 'Zurich'],axis = 1)

df_new2
plt.figure(figsize=(10, 16))

sns.heatmap(df_new2.corr().abs(), vmin=0, vmax=1, annot = True, cmap="YlGnBu")
# Теперь выделим тестовую часть

train_data = df_new.query('Sample == 1').drop(['Sample'], axis=1)

test_data = df_new.query('Sample == 0').drop(['Sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    

regr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

regr.fit(X_train, y_train)  



y_pred = regr.predict(X_test)    

    

mea = metrics.mean_absolute_error(y_pred, y_test)

print(f'MAE: {round(mea,7)}')
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(regr.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = regr.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)