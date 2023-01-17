# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import datetime



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
import pandas as pd

import re

#import math

import numpy as np

from datetime import datetime

from datetime import timedelta

import json



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from sklearn.model_selection import train_test_split
#на всякий случай, заново подгружаем данные

DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')



df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data.info()
def numb_of_rev(row):

    if pd.notna(row['Number of Reviews']):

        return row['Number of Reviews']

    

    if row['Rating'] == 5.0:

        return 6.0



    if row['Rating'] == 4.0:

        return 53.0



    if row['Rating'] == 3.0:

        return 22.0



    if row['Rating'] == 2.0:

        return 9.0



    if row['Rating'] == 1.0:

        return 3.0

    

    return 0.0



def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number of Reviews'] = df_output.apply(numb_of_rev, axis=1)

    

    # тут ваш код по обработке NAN

    # ....

    

    

    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    #df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    

    #cities_le = LabelEncoder()

    #cities_le.fit(df_output['City'])

    #df_output['City Code'] = cities_le.transform(df_output['City'])

    

    # Price

    df_output['Price Range'] = df_output['Price Range'].apply(lambda x: len(re.search('(\$+)$', x).group(1)) if not pd.isna(x) else np.nan)

    df_output['Price Range'] = df_output['Price Range'].fillna(df_output['Price Range'].median())



    # Cusine Style

    df_output['Cuisine Style'] = df_output['Cuisine Style'].fillna('["Vegetarian Friendly"]')

    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(lambda x: json.loads(x.replace("'", '"')))

    

    cuisine_styles = df_output['Cuisine Style'].tolist()

    cuisine_styles_count = [len(cs) for cs in cuisine_styles]

    cuisine_styles_flatten = [c for cs in cuisine_styles for c in cs]

    cuisine_styles_unq = list(set(cuisine_styles_flatten))

    

    #df_output['Cuisine Style (Count)'] = pd.Series(cuisine_styles_count)

    

    #for cuisine in cuisine_styles_unq:

    #    df_output['Cuisine Style (Has' + cuisine + ')'] = df_output['Cuisine Style'].apply(lambda x: 1 if cuisine in x else 0)

    

    # normalize Ranking

    max_rank_in_city = df_output.groupby('City')['Ranking'].max()

    k_by_london = max_rank_in_city.apply(lambda x: max_rank_in_city['London'] / x)

    

    df_output['Ranking (Normalize)'] = df_output.apply(lambda row: row['Ranking'] * k_by_london[row['City']], axis=1)

    

    # ################### 4. Feature Engineering ####################################################

    df_output['Cuisine Style (Count)'] = df_output['Cuisine Style'].apply(len)

    df_output['Ranking (Cuisine)'] = df_output.apply(lambda x: x['Ranking (Normalize)'] / 200 if x['Cuisine Style (Count)'] <= 3 and x['Ranking (Normalize)'] <= 12000 else x['Ranking (Normalize)'], axis=1)

    

    df_output['Reviews (Len)'] = data['Reviews'].fillna('[[], []]').apply(lambda x: len(x))

    

    #df_output['letter_o'] = df_output['Reviews'].fillna('[[]]').apply(lambda x: x.count('o'))

    

    #df_output['good'] = df_output['Reviews'].fillna('[[]]').apply(lambda x: x.count('good'))



    #df_output['nice'] = df_output['Reviews'].fillna('[[]]').apply(lambda x: x.count('nice'))



    #df_output['awesome'] = df_output['Reviews'].fillna('[[]]').apply(lambda x: x.count('awesome'))



    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)



    # Remove 5.0

    #df_output = df_output[df_output['Rating'] <= 4.5]

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    df_output.drop('Ranking', axis = 1, inplace=True)

    df_output.drop('Ranking (Normalize)', axis = 1, inplace=True)



    

    

    return df_output
df_preproc = preproc_data(data)

df_preproc.sample(10)
df_preproc[df_preproc['Rating'] != 0].corr()
sns.heatmap(df_preproc[df_preproc['Rating'] != 0].corr())
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

feat_importances.nlargest(45).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
def myround(x, prec=2, base=.5):

  return round(base * round(float(x)/base),prec)



predict_submission = np.array([myround(x) for x in predict_submission])
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)