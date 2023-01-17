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

data['Number of Reviews'].fillna(0, inplace=True)



def is_cap(s):

    if s in caps:

        return 1

    else:

        return 0

s = "Kabul		Tirana		Algiers		Washington		Andorra la Vella		Luanda		Buenos Aires		Yerevan		Canberra		Vienna		Baku		Manama		Dhaka		Bridgetown		Minsk		Brussels		Porto-Nova		Thimphu		La Paz		Sarajevo		Gaborone		Brasilia		Sofia		Ouagadougou		Bujumbura		Phnom Penh		Yaounde		Ottawa		Bangui		N'Djamena		Santiago		Beijing		Bogota		Brazzaville		San Jose		Zagreb		Havana		Nicosia		Prague		Copenhagen		Santo Domingo		Quito		Cairo		San Salvador		London		Tallinn		Addis Ababa		Helsinki		Paris		Tbilisi		Berlin		Athens		Guatemala		Conakry		Georgetown		Port-au-Prince		Tegucigalpa		Budapest		Reykjavik		New Delhi		Jakarta		Tehran		Baghdad		Dublin		Jerusalem		Rome		Tokyo		Kingston		Amman		Nairobi		Kuwait		Bishkek		Vientiane		Beirut		Tripoli		Vilnius		Luxembourg			Skopje		Antananarivo		Kuala Lumpur		Male		Valletta		Port Louis		Mexico		Chisinau		Monaco		Ulaanbaatar		Rabat		Naypyidaw		Kathmandu		Amsterdam		Wellington		Managua		Niamey		Abuja		Pyongyang		Oslo		Muscat		Islamabad		Panama		Asuncion		Lima		Manila		Warsaw		Lisbon		Doha		Bucharest		Moscow		Riyadh		Edinburgh		Dakar		Belgrade		Singapore		Bratislava		Ljubljana		Pretoria, Bloemfontein, Cape Town		Seoul		Madrid		Sri Jayawardenapura Kotte		Khartoum		Stockholm		Bern		Damascus		Taipei		Dushanbe		Bangkok		Lhasa		Tunis		Ankara		Ashgabat		Kiev		Abu Dhabi		Montevideo		Tashkent		Caracas		Hanoi		Cardiff		Sana’a		Lusaka		Kinshasa	Harare		Nouakchott"

caps = s.replace('\t\t','\t')

caps = s.replace('\t\t','\t')

caps = s.replace('\t\t','\t')

caps = s.replace('\t\t','\t')

caps = s.replace('\t\t','\t')

caps = s.replace('\t\t','\t')

caps = s.split('\t')

caps = list(filter(None,caps))

data['cap']= data.City.apply(is_cap)  



data.nunique(dropna=False)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

#data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data.head(5)
data.sample(5)
data['Price Range'].value_counts()
# Ваша обработка 'Price Range'

def prc_catmid(s):

    if s =='$$ - $$$':

        return 1

    else:

        return 0

def prc_catlow(s):

    if s =='$':

        return 1

    else:

        return 0    

def prc_catmax(s):

    if s =='$$$$':

        return 1

    else:

        return 0 

def prc_cat(s):

    if s =='$':

        return 1

    elif s =='$$ - $$$':

        return 2    

    elif s =='$$$$':

        return 3  

    else:

        return 0 

data['Price_mid'] = data['Price Range'].apply(prc_catmid)

data['Price_low'] = data['Price Range'].apply(prc_catlow)

data['Price_max'] = data['Price Range'].apply(prc_catmax)

data['Price_rng'] = data['Price Range'].apply(prc_cat)
# тут ваш код на обработку других признаков

# .....

import re

pattern = re.compile('[0-3][0-9]/[0-3][0-9]/[1-2][09][0-9][0-9]')



def is_cus(s):

    if str(cus) in str(s):

        return 1

    else:

        return 0

def cus_cnt(s):

    if not ',' in str(s):

        return 1

    else:

        return len(str(s).split(','))   

def take_date1(s):

    dates = pattern.findall(str(s))

    if len(dates) > 0 :

        return dates[0]

    else :

        return '01/01/1900'  

def take_date2(s):

    dates = pattern.findall(str(s))

    if len(dates) > 1:

        return dates[1]

    else :

        return '01/01/1900' 

def set_dz0(x):

    if x.days > 30000:

        return 0

    else :

        return x.days     

def num_rev(x,y,z):

    if z > 0 :

        return z

    else:

        return round(data[(data['Price_rng'] == x) & (data['City'] == y)& (data['Number of Reviews']!=0)]['Number of Reviews'].mean()) 

  

style = data['Cuisine Style'].str[2:-2].str.split("', '").dropna(0)

style_set = set()

for cuisine in style:

     style_set.update(cuisine)

for cus in style_set:

    data[cus] = data['Cuisine Style'].apply(is_cus)

    

data['cus_cnt'] = data['Cuisine Style'].apply(cus_cnt)

data['cus_cnt'].fillna(1)

data['last_rew']=data.Reviews.apply(take_date1)

data['last_rew'] = pd.to_datetime(data['last_rew'])  

data['pr_rew']=data.Reviews.apply(take_date2)

data['pr_rew'] = pd.to_datetime(data['pr_rew'])

data['dayz'] = data['last_rew'] - data['pr_rew']

data['dayz'] = data['dayz'].apply(set_dz0)

now = data['last_rew'].max()

data['dayztoday'] = now - data['last_rew']

data['dayztoday'] = data['dayztoday'].apply(set_dz0)

data['Number of Reviews'] = data['Number of Reviews'].fillna(0)

data['NumRev'] = data.apply(lambda x :num_rev(x['Price_rng'],x['City'],x['Number of Reviews']), axis =  1)

#data = data.drop(columns=['City', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA','last_rew','pr_rew'])

# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
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

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    df_output['Number of Reviews'].fillna(0, inplace=True)

    # тут ваш код по обработке NAN

    # ....

    

    # ################### 3. Encoding ##############################################################  

    # после 4

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    # ....

    def is_cap(s):

        if s in caps:

            return 1

        else:

            return 0

    s = "Kabul		Tirana		Algiers		Washington		Andorra la Vella		Luanda		Buenos Aires		Yerevan		Canberra		Vienna		Baku		Manama		Dhaka		Bridgetown		Minsk		Brussels		Porto-Nova		Thimphu		La Paz		Sarajevo		Gaborone		Brasilia		Sofia		Ouagadougou		Bujumbura		Phnom Penh		Yaounde		Ottawa		Bangui		N'Djamena		Santiago		Beijing		Bogota		Brazzaville		San Jose		Zagreb		Havana		Nicosia		Prague		Copenhagen		Santo Domingo		Quito		Cairo		San Salvador		London		Tallinn		Addis Ababa		Helsinki		Paris		Tbilisi		Berlin		Athens		Guatemala		Conakry		Georgetown		Port-au-Prince		Tegucigalpa		Budapest		Reykjavik		New Delhi		Jakarta		Tehran		Baghdad		Dublin		Jerusalem		Rome		Tokyo		Kingston		Amman		Nairobi		Kuwait		Bishkek		Vientiane		Beirut		Tripoli		Vilnius		Luxembourg			Skopje		Antananarivo		Kuala Lumpur		Male		Valletta		Port Louis		Mexico		Chisinau		Monaco		Ulaanbaatar		Rabat		Naypyidaw		Kathmandu		Amsterdam		Wellington		Managua		Niamey		Abuja		Pyongyang		Oslo		Muscat		Islamabad		Panama		Asuncion		Lima		Manila		Warsaw		Lisbon		Doha		Bucharest		Moscow		Riyadh		Edinburgh		Dakar		Belgrade		Singapore		Bratislava		Ljubljana		Pretoria, Bloemfontein, Cape Town		Seoul		Madrid		Sri Jayawardenapura Kotte		Khartoum		Stockholm		Bern		Damascus		Taipei		Dushanbe		Bangkok		Lhasa		Tunis		Ankara		Ashgabat		Kiev		Abu Dhabi		Montevideo		Tashkent		Caracas		Hanoi		Cardiff		Sana’a		Lusaka		Kinshasa	Harare		Nouakchott"

    caps = s.replace('\t\t','\t')

    caps = s.replace('\t\t','\t')

    caps = s.replace('\t\t','\t')

    caps = s.replace('\t\t','\t')

    caps = s.replace('\t\t','\t')

    caps = s.replace('\t\t','\t')

    caps = s.split('\t')

    caps = list(filter(None,caps))

    data['cap']= data.City.apply(is_cap) 

    

    # Ваша обработка 'Price Range'

    def prc_catmid(s):

        if s =='$$ - $$$':

            return 1

        else:

            return 0

    def prc_catlow(s):

        if s =='$':

            return 1

        else:

            return 0    

    def prc_catmax(s):

        if s =='$$$$':

            return 1

        else:

            return 0 

    def prc_cat(s):

        if s =='$':

            return 1

        elif s =='$$ - $$$':

            return 2    

        elif s =='$$$$':

            return 3  

        else:

            return 0 

    data['Price_mid'] = data['Price Range'].apply(prc_catmid)

    data['Price_low'] = data['Price Range'].apply(prc_catlow)

    data['Price_max'] = data['Price Range'].apply(prc_catmax)

    data['Price_rng'] = data['Price Range'].apply(prc_cat)    

    

    import re

    pattern = re.compile('[0-3][0-9]/[0-3][0-9]/[1-2][09][0-9][0-9]')



    def is_cus(s):

        if str(cus) in str(s):

            return 1

        else:

            return 0

    def cus_cnt(s):

        if not ',' in str(s):

            return 1

        else:

            return len(str(s).split(','))   

    def take_date1(s):

        dates = pattern.findall(str(s))

        if len(dates) > 0 :

            return dates[0]

        else :

            return '01/01/1900'  

    def take_date2(s):

        dates = pattern.findall(str(s))

        if len(dates) > 1:

            return dates[1]

        else :

            return '01/01/1900' 

    def set_dz0(x):

        if x.days > 30000:

            return 0

        else :

            return x.days     

    def num_rev(x,y,z):

        if z > 0 :

            return z

        else:

            return round(data[(data['Price_rng'] == x) & (data['City'] == y)& (data['Number of Reviews']!=0)]['Number of Reviews'].mean()) 



    data['cus_cnt'] = data['Cuisine Style'].apply(cus_cnt)

    data['cus_cnt'].fillna(1)

    data['last_rew']=data.Reviews.apply(take_date1)

    data['last_rew'] = pd.to_datetime(data['last_rew'])  

    data['pr_rew']=data.Reviews.apply(take_date2)

    data['pr_rew'] = pd.to_datetime(data['pr_rew'])

    data['dayz'] = data['last_rew'] - data['pr_rew']

    data['dayz'] = data['dayz'].apply(set_dz0)

    now = data['last_rew'].max()

    data['dayztoday'] = now - data['last_rew']

    data['dayztoday'] = data['dayztoday'].apply(set_dz0)

    data['Number of Reviews'] = data['Number of Reviews'].fillna(0)

    data['NumRev'] = data.apply(lambda x :num_rev(x['Price_rng'],x['City'],x['Number of Reviews']), axis =  1)

    #data = data.drop(columns=['City', 'Cuisine Style', 'Price Range', 'Reviews', 'URL_TA', 'ID_TA','last_rew','pr_rew'])

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na



    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    # тут ваш код не Encoding фитчей

    # ....

    style = data['Cuisine Style'].str[2:-2].str.split("', '").dropna(0)

    style_set = set()

    for cuisine in style:

         style_set.update(cuisine)

    for cus in style_set:

        data[cus] = data['Cuisine Style'].apply(is_cus)    

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