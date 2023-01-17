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
import re

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

DATA_DIR2 = '/kaggle/input/rds1-ivtara/'

df_train = pd.read_csv(DATA_DIR+'main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'sample_submission.csv')
df_train.info()
df_train.head(5)
df_train.head(5)
df_test.head(5)
sample_submission.head(5)

sample_submission.info()

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.sample(5)

data.info()
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number of Reviews'].fillna(0, inplace=True)

    # тут ваш код по обработке NAN

    # ....

    

    

    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    # тут ваш код не Encoding фитчей

    # ....

    

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    # ....

    

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    

    return df_output
# df_preproc = preproc_data(data)

# df_preproc.sample(10)



df = data.copy()

df
df=df.fillna(value={'Cuisine Style':""})

df['Cuisine Style']=df['Cuisine Style'].str.replace('[','')

df['Cuisine Style']=df['Cuisine Style'].str.replace(']','')

df['Cuisine Style']=df['Cuisine Style'].str.replace(' ','')

xr = df['Cuisine Style'].str.split(',')



flat_list = []

for sublist in xr:

    for item in sublist:

        if item!='':

            flat_list.append(item)



pd.Series(flat_list).describe()

df
df=df.fillna(value={'Cuisine Style':""})

df['Cuisine Style']=df['Cuisine Style'].tolist()

df['Cuisine Style']=df['Cuisine Style'].str.replace('[','')

df['Cuisine Style']=df['Cuisine Style'].str.replace(']','')

df['Cuisine Style']=df['Cuisine Style'].str.replace(' \'','')

df['Cuisine Style']=df['Cuisine Style'].str.replace('\' ','')

df['Cuisine Style']=df['Cuisine Style'].str.replace('\'','')



xr = df['Cuisine Style'].str.split(',')

#display(xr)









flat_list = []

for sublist in xr:

    

    for item in sublist:

        

        if item!='':

            

            flat_list.append(item)

#print(flat_list)

#pd.Series(flat_list).describe()

pd.Series(flat_list).value_counts().tail(6)

df
capitals=['Paris', 'Stockholm', 'London', 'Berlin', 'Bratislava', 'Vienna', 'Rome', 'Madrid',

       'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen',

       'Amsterdam', 'Lisbon', 'Prague', 'Oslo',

       'Helsinki', 'Edinburgh', 'Ljubljana', 'Athens',

       'Luxembourg']

#df['city is capital']=df.City.apply(lambda x: True if x in capitals else False )

#df[df['city is capital']==True]['Restaurant_id'].describe()



df['city is capital'] = df.City.apply(lambda x: 1 if x in capitals else 0 )

df
df['StyleCount']=df['Cuisine Style'].apply(lambda x: len(x.split(',')))

df
#Делим регулярным выражением, получаем серию

df['Reviews2']=df['Reviews'].str.replace(r'[\[\]]', '')

#df2 = df['Reviews2'].str.split(r'(\'.*?\')', expand=True)

#df2 = df['Reviews'].str.split(r'(\d\d/\d\d/\d\d\d\d)', expand=True)

#Слепляем

# dt=pd.concat([df, df2[5],df2[7]], axis=1)



df2 = df['Reviews2'].str.split(r'\'(\d\d/\d\d/\d\d\d\d)\'', expand=True)



#display(df2)



#Слепляем

dt=pd.concat([df, df2[1],df2[3], df2[0]], axis=1)



# values = {1:'\'01/01/2000\'', 3:'\'01/01/2000\''}

values = {1:'', 3:''}

dt=dt.fillna(value=values)

#display(dt)

#Убираем апострофы

import re 

#repl = lambda m: m.group(0)[::-1]

# def repl(arg, num=0):

#     result=''

    

#     text = arg

#     #print(re.sub(r'(\d\d)/(\d\d)/(\d{4})', r'\2.\1.\3', text)) 

#     text = re.sub(r'[\[\]]', '', text)

#     text=text.split(',')

#     if len(text)==2:

        

#         return text

#     else:

#         return text[3]

    

    

# dt = df['Reviews'].apply(repl)

# dt



# dt[5]=dt[5].str.replace('\'','')

# dt[5]=dt[5].str.replace(',','')

# dt[5]=dt[5].str.replace(' ','')

# dt[5]=dt[5].str.replace('\]\[','')



# dt[7]=dt[7].str.replace('\'','')

# dt[7]=dt[7].str.replace(',','')

# dt[7]=dt[7].str.replace(' ','')









dt[1] =  pd.to_datetime(dt[1])

dt[3] =  pd.to_datetime(dt[3])

mindate=''

if dt[1].min()>dt[3].min():

    mindate=dt[3].min()

    

else:

    mindate=dt[1].min()



#print(mindate)





maxdate=''

if dt[1].max()<dt[3].max():

    maxdate=dt[3].max()

    

else:

    maxdate=dt[1].max()



#print(maxdate)





#print(maxdate-mindate)



#Превращаем дни в цифру

dt['date diff']=abs(dt[1]-dt[3])

dt['date diff']=dt['date diff'].apply(lambda x:x.days)



# type(dt[1])

# for i in dt[1]:

#      print (i)

#dt['date diff'].max()

dt
def pricenum(arg):

    result=0

    if arg=='$$ - $$$':

        result=2

    elif arg=='$$$$':

        result=3

    elif arg=='$':

        result=1

    else:

        result=0

    

    return result







df['PriceRangeNumber']=df['Price Range'].apply(pricenum)

df
df=dt



# df.groupby('City').Ranking.max()



xt = df.groupby('City').Restaurant_id.count()

xt.name = 'Count rest'

xt = xt.to_frame()



rt = pd.merge(df, xt, on='City')

rt['Ranking2']=rt['Ranking']/rt['Count rest']

rt.head(5)



df=rt

df
xe = df.groupby('City').Ranking.max()

xe.name = 'Ranking max'

xe = xe.to_frame()

rt = pd.merge(df, xe, on='City')

rt['Ranking max']=rt['Ranking max']/rt['Ranking']

df=rt
df
#Спасибо https://stackoverflow.com/questions/29034928/pandas-convert-a-column-of-list-to-dummies

s = df['Cuisine Style'].str.split(',')

dummy = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)



# df = pd.concat([df,dummy], axis=1)

# df.head(5)
# #Мерджим со странами по названию города

# #

# from io import StringIO

# import requests



# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}

# link = 'https://datahub.io/core/world-cities/r/world-cities.csv'

# link2= 'https://public.opendatasoft.com/explore/dataset/worldcitiespop/download/?format=csv&timezone=Europe/Minsk&lang=en&use_labels_for_header=true&csv_separator=%3B'



# s=requests.get(link, headers= headers).text



# countrys=pd.read_csv(StringIO(s), sep=",")

# countrys.drop(['subcountry','geonameid'],  axis=1, inplace=True)

# countrys.columns = [ 'City', 'Country']

# countrys
#Мерджим со странами по названию города



countrys_df=pd.read_csv('/kaggle/input/tripadvisor/city.csv', sep=";")

# countrys.drop(['subcountry','geonameid'],  axis=1, inplace=True)

# countrys.columns = [ 'City', 'Country']

countrys = countrys_df[['Name','Country', 'Population']]

countrys = countrys.sort_values('Population',ascending=False)

countrys['Name'] = countrys['Name'].str.replace('ó','o')



countrys[countrys['Name']=='London']

# countrys

countrys = countrys_df[['Name','Country','Population']]

countrys = countrys.drop_duplicates('Name', keep='last')

countrys.columns = [ 'City', 'Country','Population']

countrys['City'] = countrys['City'].str.replace('ó','o')

countrys[countrys['City']=='London']

# countrys[countrys['Country']=='Poland']

# countrys
df = pd.merge(df, countrys, how='left', on='City')

df
# df[df['City']=='Krakow']
# df['City'].value_counts()
# s = df['Country']

# dummy = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)

df = pd.get_dummies(df, columns=[ 'Country',], dummy_na=True)

df
# df = pd.concat([df,dummy], axis=1)

# df
# s = df['City']

# dummy = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)

df = pd.get_dummies(df, columns=[ 'City',], dummy_na=True)

df
df = pd.concat([df,dummy], axis=1)

df
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)

#Особая замена NaN

values = {'Cuisine Style': 0, 'Ranking': df['Ranking'].mean(), 'Price Range': 0, 'Number of Reviews': df['Number of Reviews'].mean(), 'date diff':0, 'Population': df['Population'].mean()}

#values = {'Cuisine Style': 0, 'Ranking': df['Ranking'].mean(), 'Price Range': 0, 'Number of Reviews': 0}

df_fillna = df

df_fillna[['date diff', 'Number of Reviews']] = df_fillna[['date diff', 'Number of Reviews']].fillna(0)

df_fillna[['Population']]=df_fillna[['Population']].fillna(df_fillna['Population'].mean())

df_fillna.fillna(value=values, inplace = True)

df_fillna.head(5)
# drop_val = ['Restaurant_id', 'City','Cuisine Style','Price Range', 'Reviews' ,'URL_TA', 'ID_TA', 'Reviews2',1,3,0, 'Country']

drop_val = ['Restaurant_id', 'Cuisine Style','Price Range', 'Reviews' ,'URL_TA', 'ID_TA', 'Reviews2',1,3,0]



df_preproc = df_fillna.drop(drop_val, axis=1)

df_preproc.info()
df_preproc
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
X_train.info(max_cols=186)

# X_train
X_test
# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
test_data
sample_submission

predict_submission = model.predict(test_data)

predict_submission

len(predict_submission)
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)
import pandas as pd

city = pd.read_csv("../input/city.csv")

kaggle_task = pd.read_csv("../input/kaggle_task.csv")

main_task = pd.read_csv("../input/main_task.csv")
import pandas as pd

city = pd.read_csv("../input/tripadvisor/city.csv")

kaggle_task = pd.read_csv("../input/tripadvisor/kaggle_task.csv")

main_task = pd.read_csv("../input/tripadvisor/main_task.csv")

sample_submission = pd.read_csv("../input/tripadvisor/sample_submission.csv")
import pandas as pd

city = pd.read_csv("../input/tripadvisor/city.csv")

kaggle_task = pd.read_csv("../input/tripadvisor/kaggle_task.csv")

main_task = pd.read_csv("../input/tripadvisor/main_task.csv")

sample_submission = pd.read_csv("../input/tripadvisor/sample_submission.csv")