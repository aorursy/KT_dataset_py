import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
from catboost import CatBoostRegressor
VERSION    = 11

VAL_SIZE   = 0.33   # 33%
N_FOLDS    = 5
RANDOM_SEED = 42

# CATBOOST
ITERATIONS = 2000
LR         = 0.1

DIR_TRAIN  = '../input/parced/'
DIR_TEST   = '../input/sf-dst-car-price/'
import requests
from bs4 import BeautifulSoup
import csv
# Не смог спарсить с сайта ссылки на все модели, так как url обрезанной версии и полной - одинковый. 
# Стоит в последнем столбце span = кнопка, как ее передать в запрос - не разобрался :(
url_models_list=[
'https://auto.ru/moskva/cars/bmw/02/used/',
'https://auto.ru/moskva/cars/bmw/1er/used/',
'https://auto.ru/moskva/cars/bmw/2er/used/',       
'https://auto.ru/moskva/cars/bmw/2activetourer/used/',
'https://auto.ru/moskva/cars/bmw/2grandtourer/used/',
'https://auto.ru/moskva/cars/bmw/2000_c_cs/used/',
'https://auto.ru/moskva/cars/bmw/3er/used/',
'https://auto.ru/moskva/cars/bmw/321/used/',
'https://auto.ru/moskva/cars/bmw/326/used/',
'https://auto.ru/moskva/cars/bmw/340/used/',
'https://auto.ru/moskva/cars/bmw/4/used/',
'https://auto.ru/moskva/cars/bmw/5er/used/',
'https://auto.ru/moskva/cars/bmw/6er/used/',
'https://auto.ru/moskva/cars/bmw/7er/used/',
'https://auto.ru/moskva/cars/bmw/8er/used/',
'https://auto.ru/moskva/cars/bmw/e3/used/',
'https://auto.ru/moskva/cars/bmw/i3/used/',
'https://auto.ru/moskva/cars/bmw/i8/used/',
'https://auto.ru/moskva/cars/bmw/m2/used/',
'https://auto.ru/moskva/cars/bmw/m3/used/',
'https://auto.ru/moskva/cars/bmw/m4/used/',
'https://auto.ru/moskva/cars/bmw/m5/used/',
'https://auto.ru/moskva/cars/bmw/m6/used/',
'https://auto.ru/moskva/cars/bmw/x1/used/',
'https://auto.ru/moskva/cars/bmw/x2/used/',
'https://auto.ru/moskva/cars/bmw/x3/used/',
'https://auto.ru/moskva/cars/bmw/x3_m/used/',
'https://auto.ru/moskva/cars/bmw/x4/used/',
'https://auto.ru/moskva/cars/bmw/x4_m/used/',
'https://auto.ru/moskva/cars/bmw/x5/used/',
'https://auto.ru/moskva/cars/bmw/x5_m/used/',
'https://auto.ru/moskva/cars/bmw/x6/used/',
'https://auto.ru/moskva/cars/bmw/x6_m/used/',
'https://auto.ru/moskva/cars/bmw/x7/used/',
'https://auto.ru/moskva/cars/bmw/z1/used/',
'https://auto.ru/moskva/cars/bmw/z3/used/',
'https://auto.ru/moskva/cars/bmw/z4/used/',
'https://auto.ru/moskva/cars/bmw/z8/used/']
"""
Реально здесь парсить не будем, процесс долгий и тут почему то выдает ошибки

# найдем все страницы с авто - БМВ
all_pages_bmv = []
for model in url_models_list:
    for page in range(1,totalPages(model)):
        adress_i = model + "?page=" + str(page) + '&output_type=list'
        all_pages_bmv.append(adress_i)

# Оценим масштаб "бедствия"
len(all_pages_bmv)

"""
"""
# создадим список со всеми ссылками на каждую авто БМВ
# если интернет неустойчивый - прерывается с ошибкой
all_url_bmv = []
for page in all_pages_bmv:
    r = requests.get(page)
    soup = BeautifulSoup(r.text, 'lxml')
    urls_car = soup.find_all('a', class_ = 'Link ListingItemTitle-module__link')
    for url_car in urls_car:
        all_url_bmv.append(url_car.get('href'))
    

# Оценим масштаб "бедствия"
len(all_url_bmv)
"""

# запишем все в файл, на всякий случай. На Каггле пользуемся уже готовым -сохранненым ранее файлом
#url_list_df = pd.DataFrame(all_url_bmv)
#url_list_df.to_csv('all_urls_bmv.csv',encoding='utf-8-sig', index=False)
train = pd.read_csv(DIR_TRAIN+'final_parced_features.csv') # мой подготовленный датасет для обучения модели
test = pd.read_csv(DIR_TEST+'test.csv')
sample_submission = pd.read_csv(DIR_TEST+'sample_submission.csv')
def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
    # ################### Предобработка ############################################################## 
    # убираем не нужные для модели признаки
    for i in ['Таможня', 'Состояние', 'id']:
        if i in df_output.columns:
            df_output.drop([i], axis=1, inplace=True,)
    
    
    # ################### fix ############################################################## 
    # Переводим признаки из float в int (иначе catboost выдает ошибку)
    for feature in ['mileage', 'productionDate']:
        df_output[feature]=df_output[feature].astype('int32')
    
    
    # ################### Feature Engineering ####################################################
    # тут ваш код на генерацию новых фитчей
    # ....
    
    
    # ################### Clean #################################################### 
    # убираем признаки которые еще не успели обработать, 
    for i in ['Комплектация', 'description', 'Владение']:
        if i in df_output.columns:
            df_output.drop([i], axis=1, inplace=True,)
    
    return df_output
train_preproc = preproc_data(train)
X_sub = preproc_data(test)

for i in X_sub.columns:
    if i not in train_preproc.columns:
        X_sub.drop([i], axis=1, inplace = True)
#df = pd.concat([train_preproc, X_sub], ignore_index = True)
df = train_preproc

cat_features = ['bodyType', 'color', 'fuelType', 'vehicleTransmission', 'Привод', 'ПТС']
df.drop(['Руль'], axis = 1, inplace = True)
X_sub.drop(['Руль'], axis = 1, inplace = True)

def fuel(x):
    try:
        x = x.split(',')[0].lower()
    except:
        x = 'бензин'
    return x

df['fuelType'] = df['fuelType'].apply(fuel)
X_sub['fuelType'] = X_sub['fuelType'].apply(fuel)

def power(x):
    try:
        x = int(float(str(x).split(" ")[0]))
    except:
        x = 0
    return x

df['enginePower'] = df['enginePower'].apply(power)
X_sub['enginePower'] = X_sub['enginePower'].apply(power)

def engine(x):
    try:
        x = int(float(str(x).split(" ")[0]))
    except:
        x = 2
    return x

df['engineDisplacement'] = df['engineDisplacement'].apply(engine)
X_sub['engineDisplacement'] = X_sub['engineDisplacement'].apply(engine)

def owners(x):
    try:
        x = int(float((x[0])))
    except:
                x = 3
    return x

df['Владельцы'] = df['Владельцы'].apply(owners)
X_sub['Владельцы'] = X_sub['Владельцы'].apply(owners)

def price(x):
    try:
        x = int(float(x))
    except:
        x = 0
    return x

df['Price'] = df['Price'].apply(price)

df = df[df['Price'] != 0]


X = df_test.drop(['Price'], axis=1,)
y = df_test['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)

# Keep list of all categorical features in dataset to specify this for CatBoost
cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 3000)[0].tolist()

model = CatBoostRegressor(iterations = ITERATIONS,
                          learning_rate = LR,
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
df.info()
X_sub.info()
model.predict(X_sub)

