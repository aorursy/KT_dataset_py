# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'sample_submission.csv')



# # print('abc')



RANDOM_SEED = 42



df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

# data.info()



def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    #functions

    def transform_price_range(a):

        if a == '$':

            return 1

        elif a == '$$ - $$$':

            return 2

        elif a == '$$$$':

            return 3

        else:

            return 2



    def check_capital(a):

        capitals = ['Amsterdam', 'Athens', 'Berlin', 'Bratislava', 'Brussels', 'Budapest', 'Copenhagen', 'Dublin', 'Edinburgh', 'Helsinki', 'Lisbon', 'Ljubljana', 'London', 'Luxembourg', 'Madrid', 'Oslo', 'Paris', 'Prague', 'Rome', 'Stockholm', 'Vienna', 'Warsaw']

        if a in capitals:

    #         print(str)

            return 1

        else:

            return 0



    def count_cuisines(a):

        if pd.isna(a):

            return 1

        else:

            return (len(a[2:-2].split("', '")))



    def min_date(a):

        if not pd.isna(a):

            b = a[a[2:-2].rfind('[')+3:-2].replace('\'','')

            b = b.split(', ')



            temp_list = []

            for i in b:

                if len(i) > 0:

                    d = datetime.strptime(i, '%m/%d/%Y').date()

                    temp_list.append(d)

            if len(temp_list):

                m = min(temp_list)



                return m



    def max_date(a):

        if not pd.isna(a):

            b = a[a[2:-2].rfind('[')+3:-2].replace('\'','')

            b = b.split(', ')



            temp_list = []

            for i in b:

                if len(i) > 0:

                    d = datetime.strptime(i, '%m/%d/%Y').date()

                    temp_list.append(d)

            if len(temp_list):

                m = max(temp_list)



                return m



    #adding new features

    df_output['Price_rank'] = df_output['Price Range'].apply(transform_price_range)

    df_output['Capital_rank'] = df_output['City'].apply(check_capital)

    df_output['Count_cuisines'] = df_output['Cuisine Style'].apply(count_cuisines)



    df_output['min_date'] = df_output['Reviews'].apply(min_date)

    df_output['max_date'] = df_output['Reviews'].apply(max_date)

    df_output['diff'] = (df_output['max_date'] - df_output['min_date']).dt.days

    df_output['delay'] = (datetime.today().date() - df_output['max_date']).dt.days



    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)





    #filling na

    df_output = df_output.select_dtypes(exclude=['object'])

    df_output['Number of Reviews'].fillna(0, inplace=True)

    df_output['Price_rank'].fillna(0, inplace=True)

    # df['diff'].fillna(0, inplace=True)

    # df['delay'].fillna(0, inplace=True)



    df_output['diff_rank'] = df_output['diff'].rank(ascending = False, na_option = 'bottom')

    df_output['delay_rank'] = df_output['delay'].rank(ascending = False, na_option = 'bottom')



    #dropping columns

    df_output.drop(['diff','delay'], axis = 1, inplace=True) 

    

    return df_output



df_preproc = preproc_data(data)

# display(df_preproc.sample(10))

# df_preproc.info()



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



# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)



# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)



print('MAE:', metrics.mean_absolute_error(y_test, y_pred))



# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')



test_data = test_data.drop(['Rating'], axis=1)

# sample_submission

predict_submission = model.predict(test_data)

predict_submission



solution = df_test['Restaurant_id'].to_frame()

# display(solution.head())



# sample_submission['Rating'] = predict_submission

# sample_submission.to_csv('submission.csv', index=False)

# sample_submission.head(10)



# display(test_data)



def round_to_polovina(row):

    return (round(row*2.0)/2)



new_round = np.vectorize(round_to_polovina)

y_pred_round = new_round(model.predict(X_test))

print('MAE:', metrics.mean_absolute_error(y_test, y_pred_round))



solution['Rating'] = predict_submission

solution['Rating'] = solution['Rating'].apply(round_to_polovina)

solution.to_csv('solution.csv', index=False)

solution.head(10)



# sample_submission['Rating'] = predict_submission

# sample_submission['Rating'] = sample_submission['Rating'].apply(round_to_polovina)

# sample_submission.to_csv('submission.csv', index=False)

# sample_submission.head(10)