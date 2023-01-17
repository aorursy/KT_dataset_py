import numpy as np # линейная алгебра

import pandas as pd # работа с файлами



import re #регулярные выражения

from datetime import datetime #работа с датами



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



#вывод красивых и понятных графиков

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

import plotly.figure_factory as ff

import plotly.graph_objs as go



# Для работы в оффлайн режиме,чтобы выводились графики iplot

init_notebook_mode(connected=True)

cf.go_offline()



from sklearn.model_selection import train_test_split # специальный инструмент для разбивки

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42

# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')



#Внешние данные

df_population = pd.read_csv('/kaggle/input/population-cities-2020/population_cities_2020.csv') #https://worldpopulationreview.com/world-cities
df_train.head(10)
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
#Изменим названия столбцов для удобства

data.rename(columns={'Cuisine Style':'Cuisine_Style','Price Range':'Price_Range','Number of Reviews':'Number_of_Reviews'},inplace=True)
data.head(5)
#обработка Reviews

def proc_Reviews(line):

    line = line.replace('[','')

    line = line.replace(']','')

    line = line.replace("'",'')

    return line



#обработка Cuisine_Style

def proc_Cuisine_Style(line):

    line = line.replace('[','')

    line = line.replace(']','')

    line = line.replace("'",'')

    line = line.replace(" ",'')

    return line



#Диапазон цен в числах,целочисленное кодирование

#0 - нет сведений, 1 - низкий, 2 - средний, 3 - высокий

def filter_price_range(line):

    line = line.strip()

    if line == '$':

        return 1

    elif line == '$$ - $$$':

        return 2  

    elif line == '$$$$':

        return 3

    else:

        return 0



#Количество вхождений в комментарии ключевых хороших слов

def filter_good_comment(line):

    key_words = ['good','best','nice','better','amazing','excellent','great','wonderful']

    counter = 0

    for word in key_words:

        if word in line.lower():

            counter += 1

    return counter



#Разница в днях между датами отзывов

def get_date_difference(line):

    if len(line) != 2:

        return 0

    date1 = datetime.strptime(line[0], '%m/%d/%Y')

    date2 = datetime.strptime(line[1], '%m/%d/%Y')

    return (np.abs(date1 - date2)).days



# Показать график по value_counts 

def show(col, t):

    data[col].value_counts(ascending=True).iplot(kind='barh', title=t)

    

#Удалить dummy признаки

def del_dummy(df):

    for col in df.columns:

        if 'dum_' in col:

            df = df.drop(col,axis = 1)

    return df
# Делаем отдельную колонку с информацией о NAN для Number_of_Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number_of_Reviews']).astype('uint8')



# Заменяем в Number_of_Reviews пропуски на 0. При замене пропусков на среднее или среднее по городам - МАЕ хуже.

data['Number_of_Reviews'].fillna(0, axis = 0, inplace=True)

#среднее

#data['Number_of_Reviews'].fillna(round(data['Number_of_Reviews']),inplace = True) 

#среднее по городам

#means = round(df_all.groupby('City')['Number_of_Reviews'].mean())

#data.Number_of_Reviews.fillna(df_all.City.map(means),inplace = True) 
# Заменяем в Reviews пропуски на "[[], []]". Отдельно колонку не делаем,т.к. всего 2 NAN значения

data['Reviews'].fillna('[[], []]',inplace = True)
# Делаем отдельную колонку с информацией о NAN для Cuisine_Style

data['Cuisine_Style_isNAN'] = pd.isna(data['Cuisine_Style']).astype('uint8')

# Заменяем в Cuisine_Style пропуски на [Other].Скобки для соблюдения формата list

data['Cuisine_Style'].fillna('[Other]',inplace = True)
# Делаем отдельную колонку с информацией о NAN для Price_Range

data['Price_Range_isNAN'] = pd.isna(data['Price_Range']).astype('uint8')

# Заменяем в Price_Range пропуски на '-'

data['Price_Range'].fillna('-',inplace = True)
data.info()
data.drop(['Restaurant_id'], axis = 1,inplace = True)
print('Количество городов: ', data['City'].nunique())
show('City', 'Количество ресторанов в городах')
res_count = data['City'].value_counts()

data['Restaurants_Count'] = data['City'].map(res_count)
data['Cuisine_Style'] = data['Cuisine_Style'].apply(proc_Cuisine_Style)

data['Quan_Cuisine_Style'] = data['Cuisine_Style'].apply(lambda x:len(x))
#std нормализация

#means = data.groupby('City')['Ranking'].mean()

#std = data.groupby('City')['Ranking'].std()

#data['Ranking_std'] = (data.Ranking - data.City.map(means))/(data.City.map(std))



#minmax нормализация

min_rc = data.groupby('City')['Ranking'].min()

max_rc = data.groupby('City')['Ranking'].max()

data['Ranking_min_max'] = (data['Ranking'] - data['City'].map(min_rc))/(data['City'].map(max_rc) - data['City'].map(min_rc))



#minmax даёт результат МАЕ лучше,чем std.
data['Rel_Rank'] = data['Ranking'] / data['Restaurants_Count']
data['Price_Range'].value_counts()
data['Price_Range'] = data['Price_Range'].apply(lambda x: x.strip()) #уберём отступы

data['Price_Range_level']  = data['Price_Range'].apply(filter_price_range)
show('City', 'Количество ресторанов в городах')
quan_nor_city = data.groupby('City')['Number_of_Reviews'].sum()

data['Quan_NoR_City'] = data.City.map(quan_nor_city)
quan_nor_city.sort_values(ascending=True).iplot(kind='barh', title='Количество отзывов о ресторанах по городам')
data['Reviews'] = data['Reviews'].apply(proc_Reviews)
data['Date_of_Reviews']    = data['Reviews'].str.findall('\d+/\d+/\d+') #Даты из отзывов
data['Date_of_Reviews'].apply(lambda x: len(x)).value_counts() #Количество дат в отзывах
data['Good_Comments']      = data.Reviews.apply(filter_good_comment)

data['Date_Difference']    = data['Date_of_Reviews'].apply(get_date_difference)
data['URL_TA_code'] = data['URL_TA'].str.split('-').apply(lambda x : x[1][1:]).astype('float64')
data['ID_TA'] = data['ID_TA'].apply(lambda x: x[1:]).astype('float64')
data = data.merge(df_population[['Name','Population','Country']],how = 'left',left_on='City', right_on='Name')

data.drop(['Name'],axis = 1,inplace = True)
#отношение среднего ранга по городу к численности населения города

mean_rc = data.groupby('City')['Ranking'].mean()

data['Ranking_City_mean'] = data.City.map(mean_rc) / data.Population 



#Количество ресторанов в городе

res_count = data['City'].value_counts()

data['Restaurants_Count'] = data['City'].map(res_count)

#Относительный ранг ресторана среди всех ресторанов города

data['Rel_Rank'] = data['Ranking'] / data['Restaurants_Count']

#Отношение количества отзывов на население города

data['NoR_P'] = data['Number_of_Reviews'] / data['Population']

#Относительный ранг ресторана с учетом количества отзывов в городе по населению

data['RR_NoR_P'] = data['Rel_Rank'] * data['NoR_P'] 



#ранг ресторана относительно количества отзывов по городу

data['RR_QNC'] = data['Ranking'] / data['Quan_NoR_City']
pref = 'dum_'



#хорошие слова в комментариях

key_words = ['good','best','nice','better','amazing','excellent','great','wonderful']

for word in key_words:

    data[pref + word] = 0

    data[pref + word] = data['Reviews'].apply(lambda x: 1 if word in x.lower() else 0) 



#типы кухонь

set_cuisine = set(pd.DataFrame(data['Cuisine_Style'].str.split(',').tolist()).stack())

for cuis in set_cuisine:

    data[pref + cuis] = 0

    data[pref + cuis] = data['Cuisine_Style'].apply(lambda x: 1 if cuis in x else 0)



#города

data = pd.get_dummies(data,columns = ['City'], prefix='dum_city', dummy_na=True)

#страны

data = pd.get_dummies(data,columns = ['Country'], prefix='dum_country', dummy_na=True)
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
df_train['Rating'].value_counts(ascending=True).iplot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
data_temp = del_dummy(data)
plt.rcParams['figure.figsize'] = (20,10)

sns.heatmap(round(data_temp.drop(['sample'], axis=1),2).corr(),annot=True)
# на всякий случай, заново подгружаем данные

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

df_population = pd.read_csv('/kaggle/input/population-cities-2020/population_cities_2020.csv')



df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data.info()
#обработка Reviews

def proc_Reviews(line):

    line = line.replace('[','')

    line = line.replace(']','')

    line = line.replace("'",'')

    return line



#обработка Cuisine_Style

def proc_Cuisine_Style(line):

    line = line.replace('[','')

    line = line.replace(']','')

    line = line.replace("'",'')

    line = line.replace(" ",'')

    return line



#Диапазон цен в числах,целочисленное кодирование

#0 - нет сведений, 1 - низкий, 2 - средний, 3 - высокий

def filter_price_range(line):

    line = line.strip()

    if line == '$':

        return 1

    elif line == '$$ - $$$':

        return 2  

    elif line == '$$$$':

        return 3

    else:

        return 0



#Количество вхождений в комментарии ключевых хороших слов

def filter_good_comment(line):

    key_words = ['good','best','nice','better','amazing','excellent','great','wonderful']

    counter = 0

    for word in key_words:

        if word in line.lower():

            counter += 1

    return counter



#Разница в днях между датами отзывов

def get_date_difference(line):

    if len(line) != 2:

        return 0

    date1 = datetime.strptime(line[0], '%m/%d/%Y')

    date2 = datetime.strptime(line[1], '%m/%d/%Y')

    return (np.abs(date1 - date2)).days
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    df_output.rename(columns={'Cuisine Style':'Cuisine_Style','Price Range':'Price_Range','Number of Reviews':'Number_of_Reviews'},inplace=True)

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id'], axis = 1, inplace=True)

    

    # ################### 2. NAN ############################################################## 

    df_output['Number_of_Reviews_isNAN'] = pd.isna(df_output['Number_of_Reviews']).astype('uint8')

    df_output['Number_of_Reviews'].fillna(0, inplace=True)

    df_output['Reviews'].fillna('[[], []]',inplace = True)

    df_output['Cuisine_Style_isNAN'] = pd.isna(df_output['Cuisine_Style']).astype('uint8')

    df_output['Cuisine_Style'].fillna('[Other]',inplace = True)

    df_output['Price_Range_isNAN'] = pd.isna(df_output['Price_Range']).astype('uint8')

    df_output['Price_Range'].fillna('-',inplace = True)



    # ################### 3. Feature Engineering ####################################################

    df_output['Restaurants_Count'] = df_output['City'].map(df_output['City'].value_counts())

    df_output['Cuisine_Style'] = df_output['Cuisine_Style'].apply(proc_Cuisine_Style)

    df_output['Quan_Cuisine_Style'] = df_output['Cuisine_Style'].apply(lambda x:len(x))

    min_rc = df_output.groupby('City')['Ranking'].min()

    max_rc = df_output.groupby('City')['Ranking'].max()

    df_output['Ranking_min_max'] = (df_output['Ranking'] - df_output['City'].map(min_rc))/(df_output['City'].map(max_rc) - df_output['City'].map(min_rc))

    df_output['Rel_Rank'] = df_output['Ranking'] / df_output['Restaurants_Count']

    df_output['Price_Range'] = df_output['Price_Range'].apply(lambda x: x.strip())

    df_output['Price_Range_level']  = df_output['Price_Range'].apply(filter_price_range)

    df_output['Quan_NoR_City'] = df_output['City'].map(df_output.groupby('City')['Number_of_Reviews'].sum())

    df_output['Reviews'] = df_output['Reviews'].apply(proc_Reviews)

    df_output['Date_of_Reviews'] = df_output['Reviews'].str.findall('\d+/\d+/\d+')

    df_output['Good_Comments']  = df_output.Reviews.apply(filter_good_comment)

    df_output['Date_Difference'] = df_output['Date_of_Reviews'].apply(get_date_difference)

    df_output['URL_TA_code'] = df_output['URL_TA'].str.split('-').apply(lambda x : x[1][1:]).astype('float64')

    df_output['ID_TA'] = data['ID_TA'].apply(lambda x: x[1:]).astype('float64')

    df_output = df_output.merge(df_population[['Name','Population','Country']],how = 'left',left_on='City', right_on='Name')

    df_output.drop(['Name'],axis = 1,inplace = True)

   

    mean_rc = df_output.groupby('City')['Ranking'].mean()

    df_output['Ranking_City_mean'] = df_output['City'].map(mean_rc) / df_output.Population 

    res_count = df_output['City'].value_counts()

    df_output['Restaurants_Count'] = df_output['City'].map(res_count)

    df_output['Rel_Rank'] = df_output['Ranking'] / df_output['Restaurants_Count']

    df_output['NoR_P'] = df_output['Number_of_Reviews'] / df_output['Population']

    df_output['RR_NoR_P'] = df_output['Rel_Rank'] * df_output['NoR_P'] 

    df_output['RR_QNC'] = df_output['Ranking'] / df_output['Quan_NoR_City']

    

    # ################### 4. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    pref = 'dum_'

    key_words = ['good','best','nice','better','amazing','excellent','great','wonderful']

    for word in key_words:

        df_output[pref + word] = 0

        df_output[pref + word] = df_output['Reviews'].apply(lambda x: 1 if word in x.lower() else 0) 



    set_cuisine = set(pd.DataFrame(df_output['Cuisine_Style'].str.split(',').tolist()).stack())

    for cuis in set_cuisine:

        df_output[pref + cuis] = 0

        df_output[pref + cuis] = df_output['Cuisine_Style'].apply(lambda x: 1 if cuis in x else 0)

        

    df_output = pd.get_dummies(df_output,columns = ['City'], prefix='dum_city', dummy_na=True)

    df_output = pd.get_dummies(df_output,columns = ['Country'], prefix='dum_country', dummy_na=True)

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    

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

feat_importances.nlargest(30).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)