# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Create random seed

RND_SEED=1
samp = pd.read_csv('../input/sample_submission.csv')

store =  pd.read_csv('../input/store.csv')

test = pd.read_csv('../input/test.csv')

train =  pd.read_csv('../input/train.csv')
#Для того, чтобы одновременно менять фичи у test и train объединим датасеты

train['Set'] = 1

test['Set'] = 0

df = pd.concat([train,test])
store.info()
store.head()
df.head()
#Поля имеют два значения как "0" так и 0

df["StateHoliday"].loc[df["StateHoliday"] == "0"] = 0
#Преобразуем категориальные признаки StoreType, Assortment, StateHoliday

store['StoreType'] = store['StoreType'].astype('category').cat.codes

store['Assortment'] = store['Assortment'].astype('category').cat.codes

df["StateHoliday"] = df["StateHoliday"].astype('category').cat.codes
df['StateHoliday'].value_counts()
#Найдем среднее число продавцов и покупок по магазинам и объединим с таблицей store

average_sales_customers = train.groupby('Store')[["Sales", "Customers"]].mean()

sales_customers_df = pd.DataFrame({'Store':average_sales_customers.index,

                      'Sales':average_sales_customers["Sales"], 'Customers': average_sales_customers["Customers"]}, 

                      columns=['Store', 'Sales', 'Customers'])

store = pd.merge(sales_customers_df, store, on='Store')



store.head()
#Видно, что в таблице присутствуют нулевые признаки

store.isnull().sum()
#Удалим те магазины, который были открыты, но не торговали

df = df.loc[~((df['Open'] == 1) & (df['Sales'] == 0))]
#Сохраним айдишники закрытых магазинов, в качестве ответа им потом будут присвоены нули.

closed_ids = df["Id"][df["Open"] == 0].values
#Так как в нашей таблице иногда встречались нули, удалим их из closed_ids

closed_ids = closed_ids[~np.isnan(closed_ids)]
#Удалим закратые магазины из таблицы

df = df[df["Open"] != 0]
#Прежде чем объединять таблицу store и train преобразуем поле CompetitionOpenSinceYear

def convertCompetitionOpen(df):

    try:

        date = '{}-{}'.format(int(df['CompetitionOpenSinceYear']), int(df['CompetitionOpenSinceMonth']))

        return pd.to_datetime(date)

    except:

        return np.nan



store['CompetitionOpenInt'] = store.apply(lambda df: convertCompetitionOpen(df), axis=1).astype(np.int64)
#У таблицы store выделим основные фичи

features_store = ['Store', 'StoreType', 'Assortment', 

                  'CompetitionDistance', 'CompetitionOpenInt']

features_y = ['Sales']
df.head()
df = pd.merge(df, store[features_store], how='left', on=['Store'])
#Выделим основные фичи на основе которых будем предсказывать ответ

features_x = ['DayOfWeek','Promo', 'SchoolHoliday', 'StateHoliday', 'StoreType', 'Assortment', 

                  'CompetitionDistance', 'CompetitionOpenInt']
df[features_x].head()
df[features_x].isnull().sum()
#Всем Nan в CompetitionDistance присвоим -1

df['CompetitionDistance'] = df['CompetitionDistance'].fillna(-1)
#Проверим снова

df[features_x].isnull().sum()
df[features_x].head()
#Выделим обучающую выборку

X_train, y_train = np.array(df.loc[(df['Set'] == 1)][features_x]),np.array(df.loc[(df['Set'] == 1)][features_y])
#Преобразования для для пременения модели, понадобиться в дальнейшем.

y_train=np.ravel(y_train)
#Так как признаки до сих пор отшкалированы по-разному, то посмотрим как 

#поведет себя случайный лес, который требует минимальной предобработки данных.

#Попробуем подобрать наилучшую глубину дерева.

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
#Я не стал запускать данную ячейку, но у себя получил, что оптимальным будет max_depth = 24



#scores = []

#for d in tqdm_notebook(range(20,24)):

#    model =  RandomForestRegressor(max_depth=d)

#    scores.append(cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean())

#plt.plot(range(20,24), scores)
#Обучим нашу модель

randomForest = RandomForestRegressor(max_depth=24)

randomForest.verbose = True

randomForest.fit(X_train, y_train)
#Выделим тестовую дату

X_test=np.array(df.loc[(df['Set'] == 0)][features_x])

X_test.shape
#Предскажем результат

result = randomForest.predict(X_test)
#Первый дата фрейм - это открытые магазины

open_ids = df[df['Id'].isnull() == False & (df['Set'] == 0)]['Id']

a = pd.DataFrame({ "Id": open_ids, "Sales": result})
#Второй - закрытые, их заполним нулями

zeroes = np.zeros(closed_ids.shape)

b = pd.DataFrame({ "Id": closed_ids, "Sales": zeroes})
#Объединим, отсортируем по Id и преобразуем поле Id в int. Получим submission.csv

submission =  pd.concat([a,b], ignore_index=True)

submission.sort_values('Id', inplace=True)

submission['Id']=submission['Id'].astype(int)

submission.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').head()
#Score получился 0.16549