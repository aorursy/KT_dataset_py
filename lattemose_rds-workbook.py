# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from collections import Counter



#улучшение графики для retina-дисплея

%config InlineBackend.figure_format = 'retina'

# cnbkевое оформление графики

plt.style.use('ggplot')

# matplotlib воспроизводит без напоминаний

%matplotlib inline

# Для воспроизводимости результатов зададим:

# - общий параметр для генерации случайных чисел

RANDOM_SEED = 42
main_df = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/main_task.csv')

kaggle_df = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/kaggle_task.csv')
main_df.info()

main_df.nunique()
kaggle_df.info()

kaggle_df.nunique()
main_df['Tag'] = 1

kaggle_df['Tag'] = 0



# Недостающие данные по рейтингу в тестовой выборке заполним нулями

kaggle_df['Rating'] = 0



# Объединим датасеты в один для полного анализа по всем данным

df = pd.concat([main_df, kaggle_df], sort=False)
df['ID'] = df.Restaurant_id.apply(lambda s: s[3:len(s)])
df.info()

df.head()
df = df.drop(['Restaurant_id', 'Cuisine Style', 'Reviews', 'URL_TA', 'ID_TA'], axis=1)
plt.rcParams['figure.figsize'] = (10,7)

plt.hist(df['Rating'][df.Tag == 1].dropna(), bins = 10, align = 'right')

plt.xlabel('Score')

plt.ylabel('Count')

plt.title('Score Distribution')
plt.rcParams['figure.figsize'] = (10,7)

plt.hist(df['Ranking'].dropna(), bins=100)

plt.xlabel('Score')

plt.xlabel('Count')

plt.title('Ranking Distribution')
# Создадим справочник с указанием количества ресторанов для каждого города, присутствующего в датасете

city_population = {

    'London': 8173900,

    'Paris': 2240621,

    'Madrid': 3155360,

    'Barcelona': 1593075,

    'Berlin': 3326002,

    'Milan': 1331586,

    'Rome': 2870493,

    'Prague': 1272690,

    'Lisbon': 547733,

    'Vienna': 1765649,

    'Amsterdam': 825080,

    'Brussels': 144784,

    'Hamburg': 1718187,

    'Munich': 1364920,

    'Lyon': 496343,

    'Stockholm': 1981263,

    'Budapest': 1744665,

    'Warsaw': 1720398,

    'Dublin': 506211 ,

    'Copenhagen': 1246611,

    'Athens': 3168846,

    'Edinburgh': 476100,

    'Zurich': 402275,

    'Oporto': 221800,

    'Geneva': 196150,

    'Krakow': 756183,

    'Oslo': 673469,

    'Helsinki': 574579,

    'Bratislava': 413192,

    'Luxembourg': 576249,

    'Ljubljana': 277554

}





# Создадим новый признак

df['Population'] = df['City'].map(city_population)
# создание словаря и перевод символов в значения

d = {'$': 1, '$$ - $$$': 2, '$$$$': 3}

df['Price Range'] = df['Price Range'].map(d)

df['Price Range'] = df['Price Range'].fillna(0)

df['Number of Reviews'].fillna(0, inplace=True)
# генерирование фиктивных переменных

df = pd.get_dummies(df, columns=['City', ], dummy_na=True)
df
# создаем выборки

x = df.drop(['Rating'], axis=1)

y = df['Rating']

# создаем обучающую и тестовую выборку

mean_x = df['Ranking'].mean()

x.fillna(mean_x, inplace=True)



mean_y = df['Rating'].mean()

y.fillna(mean_y, inplace=True)





# ML



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

regr = RandomForestRegressor(n_estimators=100, bootstrap=True)

regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))