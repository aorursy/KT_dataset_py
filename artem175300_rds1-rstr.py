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
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
df_train = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/main_task.csv')

df_test = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/kaggle_task.csv')

sample_submission = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/sample_submission.csv')



# sample_submission.head()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data.head()
data.columns=['Restaurant_id', 'City', 'Cuisine_Style', 'Ranking',

       'Price_Range', 'Number_of_Reviews', 'Reviews', 'URL_TA', 'ID_TA', 'sample', 'Rating']
data.nunique()
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number_of_Reviews']).astype('uint8')
mean_val = data['Number_of_Reviews'].mean()

# print(mean_val)

data['Number_of_Reviews'].fillna(value=mean_val,inplace=True)
data=pd.get_dummies(data, columns=[ 'Price_Range','City',], dummy_na=True)

data[data['sample']==1].head()
# делаем из строки кухонь список кухонь

cuisine=data['Cuisine_Style'].str[2:-2].str.split("', '")



cui_all=[]

for cc in cuisine:

    if str(cc).lower() != 'nan':

        for ccc in cc:

            cui_all.append(ccc)

            



data['cui_s']=cuisine

data['cui_s'].fillna(value=0,axis=0,inplace=True)

data.head()
# делаем столбцы по кухням

dd=pd.get_dummies(data.cui_s.apply(pd.Series).stack()).sum(level=0)
data=pd.concat([data,dd],axis=1,sort=False)
data.info()
data.drop(['Restaurant_id', 'Reviews', 'Cuisine_Style', 'URL_TA', 'ID_TA','cui_s'], axis = 1,inplace=True)



from sklearn.model_selection import train_test_split
# Теперь выделим тестовую часть

train_data = data.query('sample == 1').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)



# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

train_data.shape, X.shape, X_train.shape, X_test.shape
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)



# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
plt.rcParams['figure.figsize'] = (10,5)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')
#Если все устраевает - готовим Submission

test_data = data.query('sample == 0').drop(['sample'], axis=1)

test_data = test_data.drop(['Rating'], axis=1)
predict_submission = model.predict(test_data)
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()