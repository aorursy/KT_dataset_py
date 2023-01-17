# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



import re 

import time

import datetime

from datetime import datetime, timedelta



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
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.head(5)
data.URL_TA[6]
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data.nunique(dropna=False)
data['scity']=data['City']
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

#data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.sample(5)
data['Price Range'].value_counts()
# Ваша обработка 'Price Range'

data.sample(5)
# тут ваш код на обработку других признаков

# .....
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
data.sample(5)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
data.sample(5)
data.sample(5)
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)
data.sample(5)
# на всякий случай, заново подгружаем данные

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data.info()
##########

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

data['Number of Reviews'].fillna(0, inplace=True)

###### добавим относительный размер города основываясь на максимальном рейтинге в каждом городе

gk=data.groupby('City')  

ss=gk.Ranking.max()

def rrr(nam):

    return ss[nam]



data['rcity']=data['City'].apply(lambda x: rrr(x))
# отнормируем ранги относително города

data['normrank'] = data['Ranking']/data['rcity']



# посмотрим на топ 10 городов

for x in (data['City'].value_counts())[0:10].index:

    data['normrank'][data['City'] == x].hist(bins=100)

plt.show()

 
#и на ранг в целом

data['normrank'].hist(bins=100)

#ранг стал равномерным во всем сете
#преобразуем данные сосписком кухонь из строки в список

data.rename(columns={'Cuisine Style': 'CuisineStyle'}, inplace=True) 

data.CuisineStyle = data.CuisineStyle.apply(lambda x: str(x).replace(

    '[', '').replace(']', '').replace('\'', '').replace(' ', '').split(','))
CuisineStyle = data['CuisineStyle'].tolist()

#создадим множество всех тпов кухонь

Cuisine = set()  # создаём пустое множество для хранения уникальных значений 

for i in CuisineStyle:  # начинаем перебор

    for j in i:

        Cuisine.add(j) # добавляем название типа кухни к множеству



# добавим столбец  количества типов кухонь

data['CuisineLen'] = data.CuisineStyle.apply(lambda x: len(x))

data['CuisineStyle'][0][0]
####### добавим признаки соответсвующие каждой кухне

def find_item(cell):

    if item in cell:

        return 1

    return 0

#dm = pd.DataFrame()

for item in Cuisine:

    data[item] = data['CuisineStyle'].apply(find_item)

data.head(2)
#заполним nan

data['Reviews'].fillna('[], []', inplace=True)
# выделим каждый одзыв е его дату в отдельные столбцы



data['Review'] = data.Reviews.apply(lambda x: str(x).split('],')[0])

data['date'] = data.Reviews.apply(lambda x: str(x).split('],')[1])

data['Review1'] = data.Review.apply(lambda x: str(x).split('\',')[0] if len(x) > 2 else '')

data['Review2'] = data.Review.apply(lambda x: str(x).split('\',')[1] if len(str(x).split('\',')) > 1 else '')

data['date1'] = data.date.apply(lambda x: str(x).split('\',')[0] if len(x) > 4 else '')

data['date2'] = data.date.apply(lambda x: str(x).split('\',')[1] if len(str(x).split('\',')) > 1 else '')

data = data.drop(['Review', 'date'], axis=1)  # удалим лишние столбцы

data['Review1'] = data['Review1'].apply(lambda x: str(x).replace(

    '[', '').replace(']', '').replace('\'', ''))

data['Review2'] = data['Review2'].apply(lambda x: str(x).replace(

    '[', '').replace(']', '').replace('\'', ''))

data['date1'] = data['date1'].apply(lambda x: str(x).replace(

    '[', '').replace(']', '').replace('\'', '').replace(' ', ''))

data['date2'] = data['date2'].apply(lambda x: str(x).replace(

    '[', '').replace(']', '').replace('\'', '').replace(' ', ''))

#посчитаем количество слов в отзывах

#data['lenReview1']=data['Review1'].apply(lambda x: len(x.split(' ')) if x!='' else 0)

#data['lenReview2']=data['Review2'].apply(lambda x: len(x.split(' ')) if x!='' else 0)



#модель ухудшилась
#попробуем количество ! в отзывах



data['lenReview1'] = data['Review1'].apply(lambda x: x.count('!'))

data['lenReview2'] = data['Review2'].apply(lambda x: x.count('!'))
#результат лучше если учитывать сумму а не два отдельных признака

data['lenW']=data['lenReview1']+data['lenReview2']

data= data.drop(['lenReview1','lenReview2'], axis = 1) #удалим лишние столбцы

#посчитаем длинну отзывов

#data['lenReview1'] = data['Review1'].apply(lambda x: len(x))

#data['lenReview2'] = data['Review2'].apply(lambda x: len(x))

#результат хуже
def ddd(x,y):

    if x !=''and y !='':

        return 2

    else:

        if x !=''or y !='':

            return 1

        else:

            return 0    

# посчитаем количество опубликованых отзывов чтобы учесть их если в столбце Number of Reviews стоит  0

data['countt'] = data.apply(lambda x: ddd(x.Review1, x.Review2),axis=1)
def ddd2(x,y):

    if x == 0:

        return y

    else:

        return x



 #если отзывы все таки есть учтем их

data.rename(columns={'Number of Reviews': 'Number_of_Reviews'}, inplace=True)

data['countt'] = data['countt'].apply(lambda x: float(x))

data['Number_of_Reviews'] =data.apply(lambda x: ddd2(x.Number_of_Reviews, x.countt), axis=1)

data = data.drop(['countt'], axis = 1) #удалим лишние столбцы
#pattern = re.compile('\d+/\d+/\d+')

#data['date1o']= data['date1'].apply(lambda x: pattern.search(x))

#data[data['date1o'].isna()==True]
# заполним пропуски в датах

data['date1'] = data['date1'].apply(lambda x: x if x!='' else '01/01/2985' ) 

data['date2'] = data['date2'].apply(lambda x: x if x!='' else '01/01/2970' ) 



data['date1'][38330]='01/01/2985' # одна строка в датасете имеет некорректную дату



#переведем даты в формат datetime

data['date1']= data['date1'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y' ))

data['date2']= data['date2'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y' ))
#добавим столбец с разницей в днях между двумя последними отзывами. пустые значения заменим средним 142

#максимальная разница в днях 3207.0 примерно 9 лет



data['days']=data['date1']-data['date2']
data['days'].max()    #все что больше 5479 нет одной или двух дат  = пустые значения заменим 

data['days'] = data['days'].apply(lambda x: x.days if x.days < 5400 and x.days > 0   else 3297)  

#mean - 145.98557721320378  median - 66.0   max - 3296

# заменим пустые на максимальное значение
#добавим признак показывающий что этих данных не было в сете

data['days_is_nan'] = data['days'].apply(lambda x: 0 if x<3297 else 1)
data[data['days']>0]['days'].max()
data[data['days']>0]['days'].hist(bins=100)
#data['days']=data['days']**(1/3)

#data[data['days']>0]['days'].hist(bins=100)
#добавим столбец с временем до последнего отзыва

def ddd3(x,y):

    if x > y:

        return x

    else:

        return y

# поместим в date1 старшую дату

data['date1'] = data.apply(lambda x: ddd3(x.date1, x.date2), axis=1)

data['days2'] = datetime(2020, 9, 9, 0, 0)-data['date1'] #09.09.2020

#добавим признак указывающий на отсутсвие данных 

#и замним строки в которых не было данных

#data['days2_is_nan'] = data['days2'].apply(lambda x: 0 if x.days > 0   else 1) #ухудшает модель

data['days2'] = data['days2'].apply(lambda x: x.days if x.days > 0   else 1061) #max =4538 mean =1159.516420383925median=1062.0

data[data['days2']>0]['days2'].hist(bins=100)
data[data['days2']>0]['days2'].mean()
data['Price Range'].fillna('0', inplace=True)

data['Price Range'].value_counts()



#посмотрим на распределение цен

def get_boxplot(column,score):

    fig, ax = plt.subplots(figsize = (14, 4))

    sns.boxplot(x=column, y=score, 

                data=data.loc[data.loc[:, column].isin(data.loc[:, column].value_counts().index[:10])],

               ax=ax)

    plt.xticks(rotation=45)

    ax.set_title('Boxplot for ' + column)

    plt.show()  

# заменим значения

def prrepl(stri):

    if stri =='$':

        return 1

    else:

        if stri =='$$ - $$$':

            return 2

        else:

            if stri =='$$$$':

                return 3

            else:

                return 0               

            

data['Price Range'] = data['Price Range'].apply(prrepl)
get_boxplot('Price Range','Rating') #MAE: 0.204844375

get_boxplot('Price Range','normrank')
#добавим признак выделяющий отсутсвующие данные

#data['Price_Range_nan'] = data['Price Range'].apply(lambda x: 1 if x==0 else 0)

#ухудшает модель
inp ={}

cc = data['City'].value_counts().index.tolist()

inp[cc[0]]=[1,8908081, 1572, 716584]

inp[cc[1]]=[1,2148327, 105.4, 357749]

inp[cc[2]]=[1,3266126, 607, 492400]

inp[cc[3]]=[0,1636762, 101.3, 492400]

inp[cc[4]]=[1,3644826, 891.68, 397000]

inp[cc[5]]=[0,1378689, 181.67, 884400]

inp[cc[6]]=[1,2870500, 1287.36, 884400]

inp[cc[7]]=[1,1301132, 500, 83620]

inp[cc[8]]=[1,505526, 100.05, 92990]

inp[cc[9]]=[1,1897491 , 414.7, 461000]

inp[cc[10]]=[1,872757 , 219.4, 344800]

inp[cc[11]]=[1,179277 , 32.61, 248700]

inp[cc[12]]=[0,1841179 , 755.09, 397000]

inp[cc[13]]=[0,1471508 , 310.71, 397000]

inp[cc[14]]=[0,506615 , 47.87, 357749]



inp[cc[15]]=[1,961609 , 188, 274800]

inp[cc[16]]=[1,1752286 , 525.14 , 63630]

inp[cc[17]]=[1,1790658 , 517 , 90800]

inp[cc[18]]=[1,1173179 , 115 , 85410]

inp[cc[19]]=[1,615993 , 86.40 , 173500]

inp[cc[20]]=[1,664046 , 412 , 95360]

inp[cc[21]]=[1,488100 , 118 ,  237618]

inp[cc[22]]=[0,428737 ,  91.88 ,  223500]

inp[cc[23]]=[0,237591 ,  41.66,  92990]

inp[cc[24]]=[0,200548 ,  15.93,  223500]

inp[cc[25]]=[0,769498 ,  327,  90800]

inp[cc[26]]=[1,673469 ,  454,  214300]

inp[cc[27]]=[1,655281,  715.48,  136800]

inp[cc[28]]=[1,437725,  368,  37090]

inp[cc[29]]=[1,115227,  51.47,  27600]

inp[cc[30]]=[1,284355,  163.8,  20200]

inp[cc[30]][0]
#добавим признак столицы

def add_stol(city):

    return inp[city][0]

    

data['is_stol']=data['City'].apply(lambda x: add_stol(x))



#население

def add_dem(city):

    return inp[city][1]

    

data['dem']=data['City'].apply(lambda x: add_dem(x))



#площадь

def add_S(city):

    return inp[city][2]

    

data['S']=data['City'].apply(lambda x: add_S(x))



#бюджеты



def add_bu(city):

    return inp[city][3]

    

data['bu']=data['City'].apply(lambda x: add_bu(x))
data['plotn'] = data['dem']/data['S']
kols = data['City'].value_counts().values

#преабразуем в словарь

kk={}

for i in range(0,31):

    kk[cc[i]] = kols[i]

    

    

#добавим как признак

def add_kols(city):

    return kk[city]

    

data['kols']=data['City'].apply(lambda x: add_kols(x))

#data['kols']=data['dem']/data['kols'] результат хуже
data['Idea'] = data['ID_TA'].apply(lambda x: float(x[1:]))



#data['Idea2'] = data['Restaurant_id'].apply(lambda x: float(x[3:])) 

#результата почти нет, не будем его использовать

cdata=pd.DataFrame()

cdata['Rating']=data['Rating']

cdata['Ranking']=data['Ranking']

cdata['Price Range']=data['Price Range']

cdata['Number_of_Reviews']=data['Number_of_Reviews']

cdata['rcity']=data['rcity']

cdata['normrank']=data['normrank']

cdata['CuisineLen']=data['CuisineLen']

cdata['lenW']=data['lenW']

cdata['days']=data['days']

cdata['days2']=data['days2']

cdata['dem']=data['dem']

cdata['S']=data['S']

cdata['bu']=data['bu']

cdata['plotn']=data['plotn']

cdata['kols']=data['kols']   

cdata['Idea']=data['Idea']                        

cdata['sample']=data['sample']   



plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(cdata.corr(),)

cdata.corr()

data[data['sample']==1]['Rating'].hist(bins=5)

#data['Number_of_Reviews'] = data['Number_of_Reviews']**(1/3) #выгавнивание распределения ни как не сказывается 

data[data['sample']==1]['Number_of_Reviews'].hist(bins=100)
get_boxplot('CuisineLen','Rating') # в целом количество кухонь не сильно влияет на оценку однако 10 явно выбивается из общей картины
len(data[data['CuisineLen']==10]) # таких всего 26 попробуем удалить

#data = data.loc[data['CuisineLen']!=10] # модель ухудшилась
len(data[data['CuisineLen']==10])
cdata[cdata['sample']==1].hist()

data[data['sample']==1]['plotn'].hist(bins=10)
#возможно значения больше 15000 выбросы. Попробуем удалить

#data = data.loc[data['plotn']<15000] #модель ухудшилась

 #mean = 4935.762228799952

#data['plotn'] = data['plotn'].apply(lambda x: x if x< 15000 else  4935.76) # модель ухудшилась

#data[data['sample']==1]['plotn'].hist(bins=10)
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA','kols'], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    

    # тут ваш код по обработке NAN

    # Для примера я возьму столбец Number of Reviews

    

    

    

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