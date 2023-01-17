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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from tqdm.notebook import tqdm

from catboost import CatBoostRegressor

from ipywidgets import IntProgress

from IPython.display import display

import re

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели



!pip install catboost

!pip install ipywidgets

!pip install shap
print('Python       :', sys.version.split('\n')[0])

print('Numpy        :', np.__version__)
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
VERSION    = 1

DIR_TRAIN  = '/kaggle/input/baselinesfdstcarpricepredictionv14/' # подключил к ноутбуку свой внешний датасет



DIR_TEST   = '/kaggle/input/sf-dst-car-price/'

VAL_SIZE   = 0.33   # 33%

N_FOLDS    = 5



# CATBOOST

ITERATIONS = 3000

LR         = 0.1
!ls ../input/

!ls -la
# Парсинг avto.ru

from random import randint

from time import sleep

import requests      

from bs4 import BeautifulSoup



# Общая функция,которая на текущей странице находит ссылки на каждое отдельное объявление



def html_avto(page):

    url = 'https://auto.ru/moskva/cars/bmw/all/?output_type=list&page={}'.format(page)

    soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text,'html.parser')

    html_avtos=soup.find_all('a', class_="Link ListingItemThumb")

    html_avtos = [i.attrs['href'] for i in html_avtos]

    sleep(randint(1,3))

    return html_avtos
# создадим общий список ссылок объявлений 

#avtos_all =[]

#for page in range (1, 100):

    #avto_all = html_avto(page)

    #avtos_all += avto_all
# Функия для сбора параметров со страницы каждого отдельного объявления, т.е. со страницы открываемой по ссылке

import re



def getProperties(soup):

    avto = soup.find_all('div', class_="CardSold__title")

    if avto == [] or None:

        # находим цену авто

        #get_avto1 = soup.find_all('meta', itemprop="price")

        get_avto1 = soup.find('span', class_='OfferPriceCaption__price')

        pattern = re.compile('\d+')

        price_avto = int(''.join(pattern.findall(get_avto1.text)))

    

 

        # находим марку авто - brand

        get_avto2 = soup.find_all('meta', itemprop="name")

        brand = get_avto2[3].attrs['content']

        # здесь же находим модель - model

        model = str(' '.join(get_avto2[0].attrs['content'].split()[1:]))

        

    

        # в другом теге так же находим целый список параметров

        get_avto3 = soup.find_all('span', class_='CardInfo__cell')

        # год выпуска авто- productionDate

        if get_avto3 != [] or None:

            production_date = int(get_avto3[1].get_text())

        # пробег в км

            pattern = re.compile('\d+')

            mileage = (''.join(pattern.findall(get_avto3[3].text)))

        # далее выводим тип кузова - bodyType

            body_type = get_avto3[5].get_text().split()[0].strip()

        # далее выводим количество дверей - numberOfDoors

            number_doors = (''.join(pattern.findall(get_avto3[5].text)))

        # цвет авто - color

            color = get_avto3[7].get_text()

        # тип двигателя - fuelType

            fuel_type = (get_avto3[9].get_text()).split('/')[-1].strip()

        # объем двигателя - engine_displacement

            pattern2 = re.compile('\d+\.\d+')

            engine_displacement = (''.join(pattern2.findall((get_avto3[9].get_text()).split('/')[0])))#.strip())))

        # мощность двтгателя - engine_power

            engine_power = (''.join(pattern.findall((get_avto3[9].get_text()).split('/')[1].strip())))

        # тип коробки -vehicle Transmission

            vehicle_transmission = get_avto3[13].get_text()

        # привод-drive

            drive = get_avto3[15].get_text()

        # руль-wheel

            wheel = get_avto3[17].get_text()

        # состояние-condition

            condition = get_avto3[19].get_text()

        # владелецы-owners

            owners = (''.join(pattern.findall(get_avto3[21].text)))

        # ПТС -pts

            pts = get_avto3[23].get_text()



            return body_type, brand, color, fuel_type, model, number_doors, production_date, vehicle_transmission, \

            engine_displacement, engine_power, mileage, drive, wheel, condition, owners, pts, price_avto

    
# Парсинг каждой ссылки объявления и запись собранных параметров в датасет

from itertools import product

from six import iteritems



#final_df = pd.DataFrame()

    

#columns = ['body_type', 'brand', 'color', 'fuel_type', 'model', 'number_doors', 'production_date', \

           #'vehicle_transmission', 'engine_displacement', 'engine_power', 'mileage', \

           #'drive', 'wheel', 'condition', 'owners', 'pts', 'price_avto']



#for url in avtos_all:

    #response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})

    #html = response.content #CONTENT

    #soup = BeautifulSoup(html,'html.parser', from_encoding='utf-8')#.encode(formatter=None)

    #avtos = getProperties(soup)

    #print (avtos)

    #if avtos != None:

        #data_row = pd.DataFrame([avtos], columns = columns)    

        #final_df = final_df.append(data_row, ignore_index=True)

    #sleep(randint(1,3))
# запишем 

#final_df.to_csv('train_BMW.csv', index=False)
train = pd.read_csv(DIR_TRAIN + 'train_BMW.csv')

train = pd.read_csv(DIR_TRAIN + 'train_BMW.csv') # мой подготовленный датасет для обучения модели

test = pd.read_csv(DIR_TEST+'test.csv')

sample_submission = pd.read_csv(DIR_TEST+'sample_submission.csv')
# Подготовим тестовый датасет



# заполним пропуски в количестве дверей в соответствии с типом кузова, в названии которого указано количество дверей

def numberOfDoors (row):

    if row['bodyType'] == 'внедорожник 5 дв.':

        return 5

    elif row['bodyType'] == 'хэтчбек 3 дв.':

        return 3

    elif row['bodyType'] == 'хэтчбек 5 дв.':

        return 5

    elif row['bodyType'] =='универсал 5 дв.':

        return 5

    elif row['bodyType'] =='седан 2 дв.':

        return 2

    else:

        return row ['numberOfDoors']

# Функция оставлена, потому в итоге этот признак удален, как излишний для модели       

#test['numberOfDoors'] = test.apply(lambda row: numberOfDoors (row), axis = 1)
# Функция  удаления ненужных столбцов, и очистки нужных)

def preproc_data(df_output):

    '''includes several functions to pre-process the predictor data.'''

    

    #df_output = df_input.copy()  

    

    

    for feature in ['name','bodyType', 'engineDisplacement', 'Владельцы', 'enginePower']:

        df_output[feature]=(df_output[feature].str.split().str.get(0))

        if feature == 'engineDisplacement':

            df_output[feature].loc[(df_output[feature] == 'undefined')] = '1.0'

        if feature == 'name': # создадим новую колонку - модель авто

            df_output['model'] = df_output[feature].str.replace(r'(xDrive|sDrive)', '', regex=True)

        if feature != 'bodyType' and feature != 'name': 

            df_output[feature] =(df_output[feature].astype('float')).round()



    #for feature in ['productionDate', 'engineDisplacement', 'Владельцы', 'enginePower', 'numberOfDoors', 'mileage']:            

        #df_output[feature] = df_output[feature].astype('int')

    

        

    # убираем не нужные для модели признаки

    df_output.drop(['modelDate', 'Таможня', 'vehicleConfiguration', 'name', 'description', 'Комплектация',\

                    'Владение', 'id'], axis=1, inplace=True,)

    

    # # Переводим признаки из float в int (иначе catboost выдает ошибку)

    for feature in df_output:

        if df_output[feature].dtype == 'float':

            df_output[feature] = df_output[feature].astype('int')

    

    

    

    # переименуем столбцы, чтобы в обоих датасетах они совпадали



    df_output.columns = ['body_type', 'brand', 'color', 'fuel_type', 'number_doors', 'production_date', \

                         'vehicle_transmission', 'engine_displacement', 'engine_power', 'mileage', \

                         'drive', 'wheel', 'condition', 'owners', 'pts', 'model']

    

    # поменяем последовательность колонок, переставим новую колонку с конца, чтобы последовательность столбцом в датасетах 

    # совпадала, а то обучающая модель не сможет сделать прогноз

    

    cols = df_output.columns.tolist()

    cols = cols[:4]+cols[-1:]+cols[4:-1]

    df_output = df_output[cols]

    

    return df_output
# Подготовим датасет, полученный при парсинге

def stack (nor):

    for i in nor:

        if i == 'model':

            nor[i] = nor[i].str.replace(r'(xDrive|sDrive|\(|\)|)', '', regex=True).str.split().str.get(-1)

        if i == 'number_doors':

            nor[i] = nor[i].fillna(4)

        if i == 'engine_displacement':

            nor.dropna(subset = [i], inplace=True)

        if nor[i].dtype == 'float':

            nor[i] = (nor[i].round()).astype('int')

    return nor
train_preproc = stack(train)

X_sub = preproc_data(test)
# Функция предобработки данных - Функция оставлена, потому в итоге этот признак излишен для модели 

def scaler(train_preproc):

            

    # Сделаем minmax нормализацию для числовых признаков, чтобы далее лучше обучить модель 

    #from sklearn.preprocessing import MinMaxScaler



    test_data = train_preproc.loc[:, ['number_doors', 'production_date', 'engine_displacement', 'engine_power', 'mileage', 'owners']] 

    scaler = MinMaxScaler()

    data = scaler.fit_transform(test_data)

    df_m3 = pd.DataFrame({'0_n':data[:,0],'1_n':data[:,1], '2_n':data[:,2], '3_n':data[:,3], '4_n':data[:,4], '5_n':data[:,5],}).round(2)



    train_preproc['number_doors'] = df_m3['0_n']

    train_preproc['production_date'] = df_m3['1_n']

    train_preproc['engine_displacement'] = df_m3['2_n']

    train_preproc['engine_power'] = df_m3['3_n']

    train_preproc['mileage'] = df_m3['4_n']

    train_preproc['owners'] = df_m3['5_n']

    

    train_preproc.dropna(inplace=True)

    

    # создадим новые числовые полиномиальные признаки возведя имеющиеся числовые признаки даты модели/production_date и 

    # пробега/mileage в 2 степень



    #from sklearn.preprocessing import PolynomialFeatures

    pf = PolynomialFeatures(2)

    data = pf.fit_transform(train_preproc[['production_date', 'mileage']])

    df_m = pd.DataFrame({'p_m_new1':data[:,2],'p_m_new2':data[:,3], 'p_m_new3':data[:,4], 'p_m_new4':data[:,5]}).round(3)

    

    

    

    train_MOD = pd.concat([train_preproc, df_m], axis=1) # объединяем с новыми числовыми признаками

    train_MOD.dropna(inplace=True)

    

    #for feature in train_MOD:

        #if train_MOD[feature].dtype == 'float':

            #train_MOD[feature] = (train_MOD[feature].round()).astype('int')

    #train_MOD.drop(['pts', 'color', 'condition', 'wheel', 'brand', 'number_doors'], axis=1, inplace=True) #удалида временно        

    

    return train_MOD
#train_preproc2 = scaler(train_preproc)

#X_sub2 = scaler(X_sub)
# Посмотрим на разброс и распределение признаков даты модели/production_date и пробега/mileage, потому что это основные признаки

# влияющие на цену авто
train_preproc['mileage'].hist()

train_preproc['mileage'].describe()


#train_preproc['production_date2'] = train_preproc['production_date'].apply(lambda w: np.log(w + 1)).round(3)

train_preproc['production_date'].hist()

train_preproc['production_date'].describe()
# Создадим новый признак прогноз цены по RandomForest, и запишем в отдельную колонку



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42

regr = RandomForestRegressor(n_estimators=100)





train_df = train_preproc.loc[:, ['production_date', 'mileage', 'price_avto']]

train_df = train_df.loc[train_df.production_date>2005] # ограничили разброс для обучения модели

train_df = train_df.loc[train_df.mileage<250000] # ограничили разброс для модели



X = train_df.drop(['price_avto'], axis=1,)

Y = train_df.price_avto.values

#.reshape((-1, 1))

#.values.reshape((-1, 1))



# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.

# Для тестирования мы будем использовать 25% от исходного датасета.

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)



# Обучаем модель на тестовом наборе данных

regr.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = regr.predict(X_test)
def mape(y_true, y_pred):

    return np.mean(np.abs((y_pred-y_true)/y_true))
print(f" MAPE: {mape(y_test, y_pred ):0.3f}")
# Создадим новый признак прогноз цены по LinearRegression, и запишем в отдельную колонку





train_df = train_preproc.loc[:, ['production_date', 'mileage', 'price_avto']]

train_df = train_df.loc[train_df.production_date>2005]

train_df = train_df.loc[train_df.mileage<250000]



X = train_df.drop(['price_avto'], axis=1,)

Y = train_df.price_avto.values





X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3)



from sklearn.linear_model import LinearRegression

myModel = LinearRegression() #Обозначаем, что наша модель - линейная регрессия

myModel.fit(X_train,Y_train) #обучаем модель на обучающих данных



y_pred2 = myModel.predict(X_test)
print(f" MAPE: {mape(Y_test, y_pred2 ):0.3f}")
test_df = train_preproc.loc[:, ['production_date', 'mileage']]

train_preproc['price_RF'] = (regr.predict(test_df).round()).astype('int')

train_preproc['price_LR'] = (myModel.predict(test_df).round()).astype('int')
test_df2 = X_sub.loc[:, ['production_date', 'mileage']]

X_sub['price_RF'] = (regr.predict(test_df2).round()).astype('int')

X_sub['price_LR'] = (myModel.predict(test_df2).round()).astype('int')
#  В итоге для улучшения показателя обучения, удалили излишние параметры



train_preproc = train_preproc.drop(['pts', 'condition', 'wheel', 'brand', 'color', 'number_doors'], axis=1)

X_sub = X_sub.drop(['pts', 'condition', 'wheel', 'brand','color', 'number_doors'], axis=1)
X = train_preproc.drop(['price_avto'], axis=1,)

y = train_preproc.price_avto.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
# чтобы не писать весь список этих признаков, просто вывел их через nunique(). и так сойдет)

X_train.nunique()
# Keep list of all categorical features in dataset to specify this for CatBoost

cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 250)[0].tolist()
model = CatBoostRegressor(iterations = ITERATIONS,

                          #learning_rate = LR,

                          random_seed = RANDOM_SEED,

                          eval_metric='MAPE',

                          custom_metric=['R2', 'MAE']

                         )

model.fit(X_train, y_train,

         cat_features=cat_features_ids,

         eval_set=(X_test, y_test),

         verbose_eval=100,

         use_best_model=True,

         plot=True

         )
model.save_model('catboost_single_model_baseline.model')
# Не плохое визуальное представление прогноза. Особенности, выдвигающие прогноз выше, показаны красным цветом, 

# а элементы, толкающие прогноз ниже, - синим цветом 

import shap



# load JS visualization code to notebook

shap.initjs()



# train XGBoost model

#X,y = shap.datasets.boston()

#model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)



# explain the model's predictions using SHAP

# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X)



# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values, X)
shap.summary_plot(shap_values, X, plot_type="bar")
predict_submission = model.predict(X_sub)

predict_submission
sample_submission = pd.read_csv(DIR_TEST+'sample_submission.csv')

sample_submission['price'] = predict_submission

sample_submission.to_csv(f'submission_v{VERSION}.csv', index=False)

sample_submission.head(10)
# Далее так же решение из базового шаблона соревнования, с тем как можно организовать обучение модели на 5 фолдах,

# с дальнейшим объединением предсказаний от каждой модели.
def cat_model(y_train, X_train, X_test, y_test):

    model = CatBoostRegressor(iterations = ITERATIONS,

                              learning_rate = LR,

                              eval_metric='MAPE',

                              random_seed = RANDOM_SEED,)

    model.fit(X_train, y_train,

              cat_features=cat_features_ids,

              eval_set=(X_test, y_test),

              verbose=False,

              use_best_model=True,

              plot=False)

    

    return(model)





def mape(y_true, y_pred):

    return np.mean(np.abs((y_pred-y_true)/y_true))
submissions = pd.DataFrame(0,columns=["sub_1"], index=sample_submission.index) # куда пишем предикты по каждой модели

score_ls = []

splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, y))



for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total=N_FOLDS,):

    # use the indexes to extract the folds in the train and validation data

    # используйте индексы для извлечения фолдов в train и проверки данных

    X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]

    # model for this fold - модель для этих фолдов

    model = cat_model(y_train, X_train, X_test, y_test,)

    # score model on test - оценка модели на тесте

    test_predict = model.predict(X_test)

    test_score = mape(y_test, test_predict)

    score_ls.append(test_score)

    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")

    # submissions

    submissions[f'sub_{idx+1}'] = model.predict(X_sub)

    model.save_model(f'catboost_fold_{idx+1}.model')

    

print(f'Mean Score: {np.mean(score_ls):0.3f}')

print(f'Std Score: {np.std(score_ls):0.4f}')

print(f'Max Score: {np.max(score_ls):0.3f}')

print(f'Min Score: {np.min(score_ls):0.3f}')
submissions.head(10)
submissions['blend'] = (submissions.sum(axis=1))/len(submissions.columns)

sample_submission['price'] = submissions['blend'].values

sample_submission.to_csv(f'submission_blend_v{VERSION}.csv', index=False)

sample_submission.head(10)