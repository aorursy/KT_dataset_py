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
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.sample(5)
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
# Заполним пропуски максимальным значением по данному id (см. пункт 2)



print(data['Number of Reviews'].isna().value_counts())

rev_max = data.groupby('Restaurant_id')['Number of Reviews'].max()



def REV_NAN_FILL(row):

    if np.isnan(row['Number of Reviews']) == True:

       row['Number of Reviews'] = rev_max[row.Restaurant_id]

    return row



data = data.apply(REV_NAN_FILL, axis = 1)



print(data['Number of Reviews'].isna().value_counts())





# Далее заполняем пропуски медианным значением столбца

data['Number of Reviews'].fillna(data['Number of Reviews'].median(), inplace=True)
data.nunique(dropna=False)
data.groupby('Restaurant_id').Rating.value_counts()
data.groupby('Restaurant_id').Ranking.value_counts()
r_mean = data.groupby('Restaurant_id').Ranking.mean()

r_median = data.groupby('Restaurant_id').Ranking.median()

r_sum = data.groupby('Restaurant_id').Ranking.sum()

r_min = data.groupby('Restaurant_id').Ranking.min()

r_max = data.groupby('Restaurant_id').Ranking.max()



data['Ranking_Mean'] = data.Restaurant_id.apply(lambda x: r_mean[x])

data['Ranking_Median'] = data.Restaurant_id.apply(lambda x: r_median[x])

data['Ranking_Sum'] = data.Restaurant_id.apply(lambda x: r_sum[x])

data['Ranking_Max'] = data.Restaurant_id.apply(lambda x: r_max[x])

data['Ranking_Min'] = data.Restaurant_id.apply(lambda x: r_min[x])



data
# Получаем список городов

print(data.City.unique())
# Сделаем из списка городов словарь CapitalCityDict. Записываем в него значение 1 если город-ключ является столицей, иначе 0.

CapitalCityDict = {'Paris': 1,

 'Stockholm': 1,

 'London': 1,

 'Berlin': 1, 

 'Munich': 0,

 'Oporto': 0, 

 'Milan': 0,

 'Bratislava': 1,

 'Vienna': 1, 

 'Rome': 1,

 'Barcelona': 0,

 'Madrid': 1,

 'Dublin': 1,

 'Brussels': 1,

 'Zurich': 0,

 'Warsaw': 1,

 'Budapest': 1, 

 'Copenhagen': 1,

 'Amsterdam': 1,

 'Lyon': 0,

 'Hamburg': 0, 

 'Lisbon': 1,

 'Prague': 1,

 'Oslo': 1, 

 'Helsinki': 1,

 'Edinburgh': 1,

 'Geneva': 0,

 'Ljubljana': 1,

 'Athens': 1,

 'Luxembourg': 1,

 'Krakow': 0       

}



data['Capital'] = data.City.apply(lambda x: CapitalCityDict[x])

data['Capital'].value_counts()
# Создаем новый столбец 'Population'

# Сделаем из списка городов словарь PopulationCityDict. Население.

PopulationCityDict = {'Paris': 2190327,

 'Stockholm': 972647,

 'London': 8908081,

 'Berlin': 3644826, 

 'Munich': 1456039 ,

 'Oporto': 237591, 

 'Milan': 1378689,

 'Bratislava': 425923,

 'Vienna': 1897491, 

 'Rome': 2875805,

 'Barcelona': 1636762,

 'Madrid': 3165541,

 'Dublin': 1173179,

 'Brussels': 179277,

 'Zurich': 428737,

 'Warsaw': 1758143,

 'Budapest': 1752286, 

 'Copenhagen': 615993,

 'Amsterdam': 857713,

 'Lyon': 506615,

 'Hamburg': 1841179, 

 'Lisbon': 505526,

 'Prague': 1301132,

 'Oslo': 673469, 

 'Helsinki': 643272,

 'Edinburgh': 488100,

 'Geneva': 200548,

 'Ljubljana': 284355,

 'Athens': 664046,

 'Luxembourg': 602005,

 'Krakow': 769498       

}



data['Population'] = data.City.apply(lambda x: PopulationCityDict[x])

#data['Population'].value_counts()
# Создаем новый столбец 'Res_Count'

# Сделаем из списка городов словарь ResCityDict. Значения - количество ресторанов в городе. 

ResCityDict = {'Paris': 17593,

 'Stockholm': 3131,

 'London': 22366,

 'Berlin': 8110, 

 'Munich': 3367,

 'Oporto': 2060, 

 'Milan': 7940,

 'Bratislava': 1331,

 'Vienna': 4387, 

 'Rome': 12086,

 'Barcelona': 10086,

 'Madrid': 11562,

 'Dublin': 2706,

 'Brussels': 3703,

 'Zurich': 1901,

 'Warsaw': 3210,

 'Budapest': 3445, 

 'Copenhagen': 2637,

 'Amsterdam': 4189,

 'Lyon': 2833,

 'Hamburg': 3501, 

 'Lisbon': 4985,

 'Prague': 5850,

 'Oslo': 1441, 

 'Helsinki': 1661,

 'Edinburgh': 2248,

 'Geneva': 1753,

 'Ljubljana': 647,

 'Athens': 2814,

 'Luxembourg': 759,

 'Krakow': 1832       

}



data['Res_Count'] = data.City.apply(lambda x: ResCityDict[x])

#data['Res_Count'].value_counts()
data['Delta'] = data['Res_Count'] - data['Ranking']

#data['Delta'].value_counts()
data['Relative'] = (data['Res_Count'] - data['Ranking']) / data['Res_Count']

data['Relative']
def CuisineListCount(row):

    string = row.loc['Cuisine Style']

    if type(string) == str:

        a = []

        for i in string.split("'"): 

            if i[0].isalpha() == False:

                continue

            else:

                a.append(i)

        row.loc['CuisineList'] = a

        row.loc['CuisineCount'] = len(a)

    else:

        row.loc['CuisineList'] = ['0']

        row.loc['CuisineCount'] = 1

    return row



data = data.apply(lambda x: CuisineListCount(x), axis = 1)
# Создаем новые столбцы 'C_'+'Название кухни' 

# Инициализируем их НУЛЯМИ

# Проставляем ЕДИНИЦЫ если в данноим расторане присутствует соответствующий тип кухни



def CreateDummyFromList(df):

    dummy_list = []

    for s in df.CuisineList:

        for i in s:

            if i not in dummy_list:

                dummy_list.append(i)

    for i in sorted(dummy_list):

        n = "C_" + str(i) # OR JUST i

        df[n] = 0

    

CreateDummyFromList(data)



def CuisineFill(row):

    for i in row.loc['CuisineList']:

        n = 'C_' + str(i)

        row[n] = 1

    return row



data = data.apply(lambda x: CuisineFill(x), axis = 1)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=['City'], dummy_na=True)
data.ID_TA.head()
data['ID_TA_N'] = data.ID_TA.apply(lambda x: int(x[1:]))
# Вариант 2

data['Price Range'].fillna('$$ - $$$') 

data = pd.get_dummies(data, columns=[ 'Price Range',], dummy_na=True)
data['Reviews'] = data['Reviews'].fillna("[]")

data['Reviews_Dates'] = data.Reviews.str.findall('\d\d/\d\d/\d\d\d\d')



data['Last_to_Now'] = None

data['Date_Delta'] = None





def DDD(row):

    if len(row['Reviews_Dates']) == 2:

        row['Last_to_Now'] = (pd.datetime.now() - pd.to_datetime(row['Reviews_Dates']).max()).days

        row['Date_Delta'] = abs((pd.to_datetime(row['Reviews_Dates'])[0] - pd.to_datetime(row['Reviews_Dates'])[1]).days)

        return row

    elif  len(row['Reviews_Dates']) == 1:

        row['Last_to_Now'] = (pd.datetime.now() - pd.to_datetime(row['Reviews_Dates'])[0]).days

        row['Date_Delta'] = None

        return row

    else:

        row['Last_to_Now'] = None

        row['Date_Delta'] = None

        return row



data = data.apply(lambda x: DDD(x), axis = 1)



data.head()
print(data['Date_Delta'].isna().value_counts())

print(data['Last_to_Now'].isna().value_counts())
# Заполним NaN минимальной разницей в днях по этому ID



dd_min = data.groupby('Restaurant_id').Date_Delta.min()

ln_min = data.groupby('Restaurant_id').Last_to_Now.min()



#data['Date_Delta'] = data.apply(lambda x: dd_min[x.Restaurant_id] if x['Date_Delta'] == np.NaN, axis = 1)

#data['Last_to_Now'] = data.Restaurant_id.apply(lambda x: ln_min[x])



def DATE_NAN_FILL(row):

    if np.isnan(row['Last_to_Now']) == True:

       row['Last_to_Now'] = ln_min[row.Restaurant_id]

    if np.isnan(row['Date_Delta']) == True:

       row['Date_Delta'] = dd_min[row.Restaurant_id]

    return row

data = data.apply(DATE_NAN_FILL, axis = 1)



# print(data['Date_Delta'].isna().value_counts())

# print(data['Last_to_Now'].isna().value_counts())
print(data['Date_Delta'].isna().value_counts())

print(data['Last_to_Now'].isna().value_counts())
data['Last_to_Now'].fillna(data['Last_to_Now'].median(), inplace = True)

data['Date_Delta'].fillna(data['Date_Delta'].median(), inplace = True)

print(data['Date_Delta'].isna().value_counts())

print(data['Last_to_Now'].isna().value_counts())
# # на всякий случай, заново подгружаем данные

# df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

# df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

# df_train['sample'] = 1 # помечаем где у нас трейн

# df_test['sample'] = 0 # помечаем где у нас тест

# df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



# data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

# data.info()
# def preproc_data(df_input):

#     '''includes several functions to pre-process the predictor data.'''

    

#     df_output = df_input.copy()

    

#     # ################### 1. Предобработка ############################################################## 

#     # убираем не нужные для модели признаки

#     df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

#     # ################### 2. NAN ############################################################## 

#     # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

#     # Для примера я возьму столбец Number of Reviews

#     df_output['Number of Reviews'].fillna(0, inplace=True)

    

#     data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

#     rev_max = data.groupby('Restaurant_id')['Number of Reviews'].max()



#     def REV_NAN_FILL(row):

#         if np.isnan(row['Number of Reviews']) == True:

#            row['Number of Reviews'] = rev_max[row.Restaurant_id]

#         return row

#     data = data.apply(REV_NAN_FILL, axis = 1)





# # Далее заполняем пропуски медианным значением столбца

# data['Number of Reviews'].fillna(data['Number of Reviews'].median(), inplace=True)

    

#     # ################### 3. Encoding ############################################################## 

#     # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

#     df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

#     # тут ваш код не Encoding фитчей

#     # ....

    

    

#     # ################### 4. Feature Engineering ####################################################

#     # тут ваш код не генерацию новых фитчей

#     # ....

    

    

#     # ################### 5. Clean #################################################### 

#     # убираем признаки которые еще не успели обработать, 

#     # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

#     object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

#     df_output.drop(object_columns, axis = 1, inplace=True)

    

#     return df_output
list(data)
object_columns = [s for s in data.columns if data[s].dtypes == 'object']

data.drop(object_columns, axis = 1, inplace=True)
data.info(max_cols = 300)
# df_preproc = preproc_data(data)

# df_preproc.sample(10)
# df_preproc.info()
# # Теперь выделим тестовую часть

# train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

# test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



# y = train_data.Rating.values            # наш таргет

# X = train_data.drop(['Rating'], axis=1)
train_data = data.query('sample == 1').drop(['sample'], axis=1)

test_data = data.query('sample == 0').drop(['sample'], axis=1)



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
def round_submission(x):

    d = []

    for i in x:

        z = i%1

        if z >= 0. and z<= 0.25:

            d.append(i//1)

        elif z>0.25 and z<0.75:

            d.append(i//1 +0.5)

        else:

            d.append(i//1 + 1)

    return d

        

round_predict_submission = round_submission(predict_submission)

# for a, b in zip(predict_submission, x):

#     print(a, b)
sample_submission['Rating'] = round_predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)