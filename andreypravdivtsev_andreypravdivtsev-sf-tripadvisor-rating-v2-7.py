# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import ast
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import MultiLabelBinarizer
import datetime
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели


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
df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
df_train.sample(5)
df_test.info()
df_test.sample(5)
sample_submission.info()
sample_submission.sample(5)
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
data['Number of Reviews'].fillna(0, inplace=True)
data.nunique(dropna=False)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.sample(5)
data['Price Range'].value_counts()
plt.rcParams['figure.figsize'] = (10,7)
df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=1000)
df_train['Ranking'][(df_train['City'] =='London')&(df_train['Ranking'] < 20)]
# посмотрим на топ 10 городов
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['Ranking'][df_train['City'] == x].hist(bins=100)
plt.show()
# как вариант, можно отнормировать ранг ресторанов по городам или на наеление города, больше ниже
mean_Ranking_on_City = df_train.groupby(['City'])['Ranking'].mean()
max_Ranking_on_City = df_train.groupby(['City'])['Ranking'].max()
df_train['max_Ranking_on_City'] = df_train['City'].apply(lambda x: max_Ranking_on_City[x])
df_train['mean_Ranking_on_City'] = df_train['City'].apply(lambda x: mean_Ranking_on_City[x])
#df_train['norm_Ranking_on_maxRank_in_City'] = (df_train['Ranking'] - df_train['mean_Ranking_on_City']) / (df_train['max_Ranking_on_City']-df_train['mean_Ranking_on_City'])
df_train['norm_Ranking_on_maxRank_in_City'] = (df_train['Ranking']) / (df_train['max_Ranking_on_City'])
# посмотрим на топ 10 городов
for x in (df_train['City'].value_counts())[0:10].index:
    df_train['norm_Ranking_on_maxRank_in_City'][df_train['City'] == x].hist(bins=100)
plt.show()
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 3].hist(bins=100)
plt.rcParams['figure.figsize'] = (15,10)
sns.heatmap(data.drop(['sample'], axis=1).corr(),)
# в лондоне и париже самые крутые рестораны
# на всякий случай, заново подгружаем данные
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
def cousine2list(in_str): # функция для предобработки сведений о кухне
    #if in_str.isnan:
    #    return np.nan
    #else:
    return list(in_str[1:-1].replace("'", '').split(", "))

# Построение графиков
def get_gr(df, col, ttl):
    df[col].value_counts(ascending=True).plot(kind='barh', title=ttl)
# fill0=1,2, 3 варианты заполнения Nan 0, mean(sample), mean
# LE=False True добавлять или нет LabelEncoder

# тут находится все обработка входящих данных
def preproc_data(df_input,fill0=0,LE=True):        
    df_output = df_input.copy()
    
    # Выполняем такие операции по всем параметрам
    # ################### 1. Предобработка ############################################################## 
    # сохраним информацию о пропусках чтобы не потерять
    # ################### 2. NAN ############################################################## 
    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...
    # ################### 3. Encoding ##############################################################  
    # ################### 4. Feature Engineering ####################################################
    # тут ваш код не генерацию новых фитчей
    
    # убираем ненужные для модели признаки
    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)
    df_output['Number of Reviews'].fillna(0, inplace=True)
    
    
    # #########################################################################################
    # обработка 'Price Range
    df_output['NaN_Price Range'] = pd.isna(df_output['Price Range']).astype('float64')   
    price_collection = {'$': 1, '$$ - $$$': 2, '$$$$': 3}
    df_output['Price Range'] = df_output['Price Range'].replace(to_replace=price_collection)    
    col='Price Range'    
    if fill0==1:
        df_output[col].fillna(0, inplace=True)
    elif fill0==2:
        mean_sample0 = df_output[df_output["sample"]==0][col].mean()
        mean_sample1 = df_output[df_output["sample"]==1][col].mean()    
        df_output.loc[(df_output[col].isna())&(df_output["sample"]==0),col]=mean_sample0
        df_output.loc[(df_output[col].isna())&(df_output["sample"]==1),col]=mean_sample1
    elif fill0==3:
        mean_sample = df_output[col].mean()
        df_output.loc[(df_output[col].isna()),col]=mean_sample
       
    # #########################################################################################
    # Cuisine Style - предобработка
    df_output['NaN_Cuisine Style'] = pd.isna(df_output['Cuisine Style']).astype('float64')
    df_output["Cuisine Style"]=df_output["Cuisine Style"].fillna("['Cuisine_no_data']")          
    
    # закодируем значения в переменной до их преобразования
    # вариант интеллектуальной кодировки, показался не нужным
    if LE==True:
        le = LabelEncoder()
        le.fit(df_output['Cuisine Style'])
        df_output['code_Cuisine Style'] = le.transform(df_output['Cuisine Style'])
                
    #преобразуем Cuisine Style в листы
    df_output["Cuisine Style"]=df_output["Cuisine Style"].apply(cousine2list)  
    # обработка 'Cuisine Style'из листов в dummies
    mlb = MultiLabelBinarizer()
    df_dummies = pd.DataFrame(mlb.fit_transform(df_output["Cuisine Style"]),columns=mlb.classes_, index=df_output.index)        
    dummies = list(df_dummies.columns)
    df_output = pd.concat([df_output, df_dummies], axis=1) 
    
    df_output["Number of Cuisines"]=df_output["Cuisine Style"].apply(lambda x: len(x))
    
    # #########################################################################################      
    # Number of Reviews обработка
    df_output['NaN_Number of Reviews'] = pd.isna(df_output['Number of Reviews']).astype('float64')    
    
    # #########################################################################################      
    # Reviews обработка
    # заменим nan на ' '
    df_output["Reviews"] = df_output["Reviews"].str.replace("nan", "' '")
    # заменим пустые строки конструкцией
    df_output["Reviews"].fillna("[[], []]", inplace=True)
    # интерпритируем строку питона
    df_output["Reviews"] = df_output["Reviews"].apply(lambda x: ast.literal_eval(x))
    # переведем текст в дату
    df_output["Reviews Dates"] = df_output["Reviews"].apply(
        lambda x: [datetime.strptime(d, "%m/%d/%Y") for d in x[1]])    
    df_output["Reviews Dates"]=df_output["Reviews Dates"].apply(lambda x: x if (len(x)>=1) else [np.nan])
    # упорядочиваем данные по убыванию
    df_output["Reviews Dates"] = df_output["Reviews Dates"].apply(
        lambda x: [x[i] for i in [1, 0]]
        if ((len(x) > 1) and (x[1] > x[0])) else x)
    df_output.drop("Reviews", axis = 1, inplace=True)
    
    
    # Reviews Dates - создание новых признаков
    # разница дней отзывов
    df_output["Reviews Dates Diff"] = df_output["Reviews Dates"].apply(
        lambda x: (x[0] - x[1]).days if len(x) > 1 else np.nan)
    
    # сохраним информацию о пропусках чтобы не потерять
    df_output['NaN_Reviews Dates Diff'] = pd.isna(df_output['Reviews Dates Diff']).astype('float64')
    
    #заменяем средними значениями пропуски в Reviews Dates Diff
    col='Reviews Dates Diff'
    if fill0==1:
        df_output[col].fillna(0, inplace=True)
    elif fill0==2:
        mean_sample0 = df_output[df_output["sample"]==0][col].mean()
        mean_sample1 = df_output[df_output["sample"]==1][col].mean()    
        df_output.loc[(df_output[col].isna())&(df_output["sample"]==0),col]=mean_sample0
        df_output.loc[(df_output[col].isna())&(df_output["sample"]==1),col]=mean_sample1
    elif fill0==3:
        mean_sample = df_output[col].mean()
        df_output.loc[(df_output[col].isna()),col]=mean_sample   
                
    # Last Review Date дата последнего отзыва относительно сегодня
    LRD = datetime(2020,10,10) # сегодня
    df_output["Last Review Date"] = df_output["Reviews Dates"].apply(
        lambda x: (LRD-x[0]).days if len(x)>1 else np.nan if (pd.isnull(x)) else (LRD-x[0]).days)                
    
    # сохраним информацию о пропусках чтобы не потерять
    df_output['NaN_Last Review Date'] = pd.isna(df_output['Last Review Date']).astype('float64')
    
    #заменяем средними значениями пропуски в Last Review Date num
    col="Last Review Date"
    if fill0==1:
        df_output[col].fillna(0, inplace=True)
    elif fill0==2:
        mean_sample0 = df_output[df_output["sample"]==0][col].mean()
        mean_sample1 = df_output[df_output["sample"]==1][col].mean()    
        df_output.loc[(df_output[col].isna())&(df_output["sample"]==0),col]=mean_sample0
        df_output.loc[(df_output[col].isna())&(df_output["sample"]==1),col]=mean_sample1
    elif fill0==3:
        mean_sample = df_output[col].mean()
        df_output.loc[(df_output[col].isna()),col]=mean_sample        
                    
    df_output.drop("Reviews Dates", axis = 1, inplace=True)
    
    # #########################################################################################      
    # Ranking обработка, стандартизация, добавление новых признаков

    mean_Ranking_on_City0 = df_output[df_output["sample"]==0].groupby(['City'])['Ranking'].mean()
    max_Ranking_on_City0 = df_output[df_output["sample"]==0].groupby(['City'])['Ranking'].max()
    count_Restorant_in_City0 = df_output[df_output["sample"]==0]['City'].value_counts(ascending=False)
    
    mean_Ranking_on_City1 = df_output[df_output["sample"]==1].groupby(['City'])['Ranking'].mean()
    max_Ranking_on_City1 = df_output[df_output["sample"]==1].groupby(['City'])['Ranking'].max()
    count_Restorant_in_City1 = df_output[df_output["sample"]==1]['City'].value_counts(ascending=False)    
    
    df_output.loc[df_output["sample"]==0,'max_Ranking_on_City'] = df_output[df_output["sample"]==0]['City'].apply(lambda x: max_Ranking_on_City0[x])
    df_output.loc[df_output["sample"]==0,'mean_Ranking_on_City'] = df_output[df_output["sample"]==0]['City'].apply(lambda x: mean_Ranking_on_City0[x])
    df_output.loc[df_output["sample"]==0,'count_Restorant_in_City'] = df_output[df_output["sample"]==0]['City'].apply(lambda x: count_Restorant_in_City0[x])    
    
    df_output.loc[df_output["sample"]==1,'max_Ranking_on_City'] = df_output[df_output["sample"]==1]['City'].apply(lambda x: max_Ranking_on_City1[x])            
    df_output.loc[df_output["sample"]==1,'mean_Ranking_on_City'] = df_output[df_output["sample"]==1]['City'].apply(lambda x: mean_Ranking_on_City1[x])
    df_output.loc[df_output["sample"]==1,'count_Restorant_in_City'] = df_output[df_output["sample"]==1]['City'].apply(lambda x: count_Restorant_in_City1[x])
    
    # #########################################################################################      
    # City обработка, добавление новых признаков            
    df_output['NaN_City'] = pd.isna(df_output['City']).astype('float64')  
    
    # добавляем информацию о городах
    dict_Сity_population= {'London' : 8908, 'Paris' : 2206, 'Madrid' : 3223, 'Barcelona' : 1620, 
                        'Berlin' : 6010, 'Milan' : 1366, 'Rome' : 2872, 'Prague' : 1308, 
                        'Lisbon' : 506, 'Vienna' : 1888, 'Amsterdam' : 860, 'Brussels' : 179, 
                        'Hamburg' : 1841, 'Munich' : 1457, 'Lyon' : 506, 'Stockholm' : 961, 
                        'Budapest' : 1752, 'Warsaw' : 1764, 'Dublin' : 553, 
                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 
                        'Zurich' : 415, 'Oporto' : 240, 'Geneva' : 201, 'Krakow' : 769, 
                        'Oslo' : 681, 'Helsinki' : 643, 'Bratislava' : 426, 
                        'Luxembourg' : 119, 'Ljubljana' : 284}
    #df_output['Сity_population'] = df_output["City"].replace(to_replace=dict_Сity_population)
    df_output['Сity_population'] = df_output["City"].apply(lambda x: dict_Сity_population[x])    
    
    # соотношение города и столицы страны
    dict_Сity_capital = {    'London': 1,    'Paris': 1,    'Madrid': 1,    'Barcelona': 0,    'Berlin': 1,
                             'Milan': 0,    'Rome': 1,    'Prague': 1,    'Lisbon': 1,    'Vienna': 1,    'Amsterdam': 1,    'Brussels': 1,
                             'Hamburg': 0,    'Munich': 0,    'Lyon': 0,    'Stockholm': 1,    'Budapest': 1,    'Warsaw': 1,    'Dublin': 1,
                             'Copenhagen': 1,    'Athens': 1,    'Edinburgh': 1,    'Zurich': 1,    'Oporto': 0,    'Geneva': 1,    'Krakow': 0,
                             'Oslo': 1,    'Helsinki': 1,    'Bratislava': 1,    'Luxembourg': 1,    'Ljubljana': 1}
    df_output['Capital'] = df_output['City'].apply(lambda x: dict_Сity_capital[x])
    
    # соотношение города и страны
    dict_Сity_coutry = {    'London': 'UK',    'Paris': 'FR',    'Madrid': 'ESP',    'Barcelona': 'ESP',    'Berlin': 'GER',
                            'Milan': 'IT',    'Rome': 'IT',    'Prague': 'Czech',    'Lisbon': 'PORT',    'Vienna': 'Austria',    'Amsterdam': 'Nederlands',
                            'Brussels': 'BELG',    'Hamburg': 'GER',    'Munich': 'GER',    'Lyon': 'FR',    'Stockholm': 'Sweden',   'Budapest': 'Hungary',
                            'Warsaw': 'PL',    'Dublin': 'Ireland',    'Copenhagen': 'Denmark',    'Athens': 'Greece',    'Edinburgh': 'Schotland',    'Zurich': 'Switzerland',
                            'Oporto': 'PORT',    'Geneva': 'Switzerland',    'Krakow': 'PL',    'Oslo': 'Norway',    'Helsinki': 'Finland',    'Bratislava': 'Slovakia',
                            'Luxembourg': 'Luxembourg',    'Ljubljana': 'Slovenija'    }
    df_output['Country'] = df_output['City'].apply(lambda x: dict_Сity_coutry[x])
    
    #колличетство ресторанов в стране
    df_tmp = df_output['Country'].value_counts()
    df_output['Rest_Count_Country'] = df_output['Country'].apply(lambda x: df_tmp[x])

    # вариант интеллектуальной кодировки, показался не нужным LE=False
    if LE==True:
        le = LabelEncoder()
        le.fit(df_output['City'])
        df_output['code_City'] = le.transform(df_output['City'])        
    
    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True, drop_first=False)                                               
    df_output = pd.get_dummies(df_output, columns=[ 'Country',], dummy_na=True, drop_first=False)                                               
   
    # ################### 4. Feature Engineering ####################################################
     
    df_output['norm_Ranking_on_max_mean'] = (df_output['Ranking'] - df_output['mean_Ranking_on_City']) / (df_output['max_Ranking_on_City']-df_output['mean_Ranking_on_City'])
    df_output['norm_Ranking_on_max'] = (df_output['Ranking']) / (df_output['max_Ranking_on_City'])    
    df_output['norm_Population_on_Rest'] = df_output['Сity_population']/df_output['count_Restorant_in_City']
    df_output['norm_Ranking_on_Population'] = (df_output['Ranking']) / (df_output['Сity_population'])
        
                    
    #PolynomialFeatures
    #df_output["Cuisines_Price"]=df_output["Number of Cuisines"]*df_output['Price Range']
    #df_output["temp2"]=df_output["Number of Reviews"]*df_output['Reviews Dates Diff']
    #df_output["temp3"]=df_output["Number of Reviews"]*df_output['Number of Cuisines']

    #pf = PolynomialFeatures(2)
    #poly_features = pf.fit_transform(df[['col1', 'col2']])
    #df_pf=pd.DataFrame(poly_features,columns=["1", "a", "b", "a^2", "ab", "b^2"])
    #df = pd.concat([df, df_pf], axis=1)

    # ################### 5. Clean #################################################### 
    Nan_cols = ['NaN_Price Range', 'NaN_Number of Reviews', 'NaN_Cuisine Style', 'NaN_City',  'NaN_Reviews Dates Diff', 'NaN_Last Review Date']
    for col in Nan_cols:
        #display(df_output[col].sum())
        if df_output[col].sum()<len(df_output)/100.0:
            df_output.drop(col, axis = 1, inplace=True)
    
    # убираем признаки которые еще не успели обработать, 
    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим
    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']
    df_output.drop(object_columns, axis = 1, inplace=True)
    
    return df_output
df_preproc = preproc_data(data,fill0=3,LE=False)
df_preproc.sample(10)
df_preproc.info()
display("number of NaNs {}".format(df_preproc.isna().sum().sum()))

df_preproc.drop("Rest_Count_Country", axis = 1, inplace=True)
#df_preproc['Last Review Date'][df_preproc['Rating'] == 4].hist(bins=100)
#get_gr(df_preproc,'Country', 'Распределение целевой переменной')
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

# Model 
# Сам ML

# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)100 1 -1
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
plt.rcParams['figure.figsize'] = (15,10)
sns.heatmap(df_preproc.drop(['sample'], axis=1).corr(),)
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
display(sample_submission)
predict_submission = model.predict(test_data)
predict_submission = np.round(predict_submission*2)/2
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)




