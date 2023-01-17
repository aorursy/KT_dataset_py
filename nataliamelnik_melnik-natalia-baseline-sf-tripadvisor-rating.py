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

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
data.Reviews[1]
# Для примера я возьму столбец Number of Reviews
data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...
data['Number of Reviews'] = data['Number of Reviews'].fillna(data['Number of Reviews'].median())

data.nunique(dropna=False)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: '\'other_style\'' if pd.isnull(x) else x[1:-1])
from collections import Counter
cuisines_count = Counter()
for cuisine in data['Cuisine Style'].str.split(', '):
    cuisines_count += Counter(cuisine)
cuisines_list = cuisines_count.most_common(20)
cuisines_top = []
for i in range (0, len(cuisines_list)):
    cuisines_top.append(cuisines_list[i][0][1:-1])
def find_item(cell):
    if item in cell:
        return 1
    return 0

for item in cuisines_top:
    data[item] = data['Cuisine Style'].apply(find_item)
data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: x.split(', ') )

data['Cuisine_counts'] = data['Cuisine Style'].apply(lambda x: len(x))
data.head(5)
data.sample(5)
data['Price Range'].value_counts()
price_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3}
data['Price Range'] = data['Price Range'].replace(to_replace=price_dict)
data['Price Range']=data['Price Range'].fillna(data['Price Range'].median())
data['rev_dates'] = data.Reviews.apply(lambda x : [0] if pd.isna(x) else x[2:-2].split('], [')[1][1:-1].split("', '"))
data['max_rev_date'] = pd.to_datetime(data['rev_dates'].apply(lambda x: max(x)))
data['first_rev'] = pd.to_datetime(data['rev_dates'].apply(lambda x : x[0]))
data['second_rev'] = pd.to_datetime(data['rev_dates'].apply(lambda x: x[1] if len(x) == 2 else ''))
data['rev_delta'] = np.abs(data['first_rev'] - data['second_rev'])
data['rev_delta'] = data['rev_delta'].apply(lambda x: x.days)
dummies = pd.get_dummies(data.City, drop_first=True)
data = pd.concat([data, dummies], axis=1) 
data.head()
data = data.drop(['Cuisine Style', 'Reviews', 'URL_TA', 'ID_TA', 'max_rev_date', 'rev_dates', 'City', 'first_rev', 'second_rev'], axis = 1)

df['rev_delta'] = df['rev_delta'].fillna(df['rev_delta'].median())
data.boxplot(column=['Number of Reviews'])
median = data['Number of Reviews'].median()
IQR = data['Number of Reviews'].quantile(0.75) - data['Number of Reviews'].quantile(0.25)
perc25 = data['Number of Reviews'].quantile(0.25)
perc75 = data['Number of Reviews'].quantile(0.75)
print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75)
      , "IQR: {}, ".format(IQR),"Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
data['Number of Reviews'].hist(bins = 50)
data['Number of Reviews'][data['Number of Reviews'] < 2000].hist(bins = 100)
data['Number of Reviews'] = data['Number of Reviews'].apply(lambda x : median if x>1000 else x)

median_dr = data['rev_delta'].median()
IQR_dr = data['rev_delta'].quantile(0.75) - data['rev_delta'].quantile(0.25)
perc25_dr = data['rev_delta'].quantile(0.25)
perc75_dr = data['rev_delta'].quantile(0.75)
print('25-й перцентиль: {},'.format(perc25_dr), '75-й перцентиль: {},'.format(perc75_dr)
      , "IQR: {}, ".format(IQR_dr),"Границы выбросов: [{f_dr}, {l_dr}].".format(f_dr=perc25_dr - 1.5*IQR_dr, l_dr=perc75_dr + 1.5*IQR_dr))
data['rev_delta'].hist(bins = 50)
data['rev_delta'] = data['rev_delta'].apply(lambda x : median_dr if x>500 else x)
data['rev_delta'] = data['rev_delta'].apply(lambda x : median_dr if pd.isna(x) else x)

df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
plt.rcParams['figure.figsize'] = (15,10)
sns.heatmap(data.drop(['sample'], axis=1).corr(),)
# на всякий случай, заново подгружаем данные
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
 ### заменяем пропуски на медианноое значение##
    df_output['Number of Reviews'] = df_output['Number of Reviews'].fillna(df_output['Number of Reviews'].median())
 
 ### заменяем пропуски на "other_style", считаем топ 20 кухонь и создаем новые признаки с назнаниями этих кухонь###
    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(lambda x: '\'other_style\'' if pd.isnull(x) else x[1:-1])
    from collections import Counter
    cuisines_count = Counter()
    for cuisine in df_output['Cuisine Style'].str.split(', '):
        cuisines_count += Counter(cuisine)
    cuisines_list = cuisines_count.most_common(20)
    cuisines_top = []
    for i in range (0, len(cuisines_list)):
        cuisines_top.append(cuisines_list[i][0][1:-1])
    
    def find_item(cell):
        if item in cell:
            return 1
        return 0

    for item in cuisines_top:
        df_output[item] = df_output['Cuisine Style'].apply(find_item)
  
 ### считаем количество кухнь у каждого ресторана и создаем новый признак###      
    df_output['Cuisine Style'] = df_output['Cuisine Style'].apply(lambda x: x.split(', ') )
    df_output['Cuisine_counts'] = df_output['Cuisine Style'].apply(lambda x: len(x))
    
  ### заменяем значения на последовательные числа###
    price_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3}
    df_output['Price Range'] = df_output['Price Range'].replace(to_replace=price_dict)
    df_output['Price Range'] = df_output['Price Range'].fillna(df_output['Price Range'].median())
  
  ### считаем количество дней между отзывами###
    df_output['rev_dates'] = df_output.Reviews.apply(lambda x : [0] if pd.isna(x) else x[2:-2].split('], [')[1][1:-1].split("', '"))
    df_output['max_rev_date'] = pd.to_datetime(df_output['rev_dates'].apply(lambda x: max(x)))
    df_output['first_rev'] = pd.to_datetime(df_output['rev_dates'].apply(lambda x : x[0]))
    df_output['second_rev'] = pd.to_datetime(df_output['rev_dates'].apply(lambda x: x[1] if len(x) == 2 else ''))
    df_output['rev_delta'] = np.abs(df_output['first_rev'] - df_output['second_rev'])
    df_output['rev_delta'] = df_output['rev_delta'].apply(lambda x: x.days)

    ### создаем dummy - признаки для городов ###
    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)
    
    ### заменяем выбросы на медианное значение ###
    median = df_output['Number of Reviews'].median()
    df_output['Number of Reviews'] = df_output['Number of Reviews'].apply(lambda x : median if x>1000 else x)
    
    ### заменяем пропуски и выбросы на медианное значение ###
    median_dr = df_output['rev_delta'].median()
    df_output['rev_delta'] = df_output['rev_delta'].apply(lambda x : median_dr if x>500 else x)
    df_output['rev_delta'] = df_output['rev_delta'].apply(lambda x : median_dr if pd.isna(x) else x)

    
    ### удаляем лишние признаки###
    df_output.drop(['Restaurant_id','Cuisine Style', 'Reviews', 'URL_TA', 'ID_TA', 'max_rev_date', 'rev_dates', 'first_rev', 'second_rev'], axis = 1, inplace=True)

    
    return df_output
df_preproc = preproc_data(data)
df_preproc.sample(10)
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
