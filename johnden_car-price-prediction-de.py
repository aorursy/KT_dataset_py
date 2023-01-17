import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
from catboost import CatBoostRegressor
import json
import matplotlib as plt
# фиксируем RANDOM_SEED!
RANDOM_SEED = 42
VERSION    = 11
DIR_TRAIN  = '../input/data-for-car-price-prediction-model/' # подключил к ноутбуку свой внешний датасет
DIR_TEST   = '../input/data-for-car-price-prediction-model/'
VAL_SIZE   = 0.33   # 33%
N_FOLDS    = 5

# CATBOOST
ITERATIONS = 2000
LR         = 0.1
bmw = pd.read_csv(DIR_TRAIN+'bmw_data.csv')
audi = pd.read_csv(DIR_TRAIN+'audi_data.csv')
medcedes = pd.read_csv(DIR_TRAIN+'MERCEDES_data.csv')
vw = pd.read_csv(DIR_TRAIN+'VOLKSWAGEN_data.csv')
big = pd.read_csv(DIR_TRAIN+'big_offers.csv')
print('bmw', bmw.shape)
print('audi', audi.shape)
print('medcedes', medcedes.shape)
print('vw', vw.shape)
print('big', big.shape)
train = bmw
train = pd.concat([train, audi], ignore_index=True)
train = pd.concat([train, medcedes], ignore_index=True)
train = pd.concat([train, vw], ignore_index=True)
train = pd.concat([train, big], ignore_index=True)
def preproc_train(data):

    # bodyType
    data['bodyType'] = data['vehicle_info'].apply(lambda x: eval(x).get('configuration').get('human_name'))

    # brand
    data['brand'] = data['vehicle_info'].apply(lambda x: eval(x).get('mark_info').get('name'))

    # color
    data['color'] = data['color_hex']

    # fuelType
    data['fuelType'] = data['lk_summary'].apply(lambda x: x.split(', ')[3])

    # modelDate
    data['modelDate'] = data['vehicle_info'].apply(lambda x: eval(x).get('super_gen').get('year_from'))

    # name
    data['name'] = data['vehicle_info'].apply(lambda x: eval(x).get('tech_param').get('human_name'))

    # numberOfDors
    data['numberOfDoors'] = data['vehicle_info'].apply(lambda x: eval(x).get('configuration').get('doors_count'))

    # productionDate
    data['productionDate'] = data['documents'].apply(lambda x: eval(x).get('year'))

    # vechileTransmission
    data['vehicleTransmission'] = data['vehicle_info'].apply(lambda x: eval(x).get('tech_param').get('transmission'))

    # enginePower
    data['enginePower'] = data['vehicle_info'].apply(lambda x: eval(x).get('tech_param').get('power'))

    # mileage
    data['mileage'] = data['state'].apply(lambda x: eval(x).get('mileage'))

    # Привод
    data['Привод'] = data['lk_summary'].apply(lambda x: x.split(', ')[2])

    # Руль
    data['Руль'] = data['vehicle_info'].apply(lambda x: eval(x).get('steering_wheel'))

    # Состояние
    data['Состояние'] = data['state'].apply(lambda x: eval(x).get('state_not_beaten'))

    # Владельцы
    data['Владельцы'] = data['documents'].apply(lambda x: eval(x).get('owners_number'))

    # ПТС
    data['ПТС'] = data['documents'].apply(lambda x: eval(x).get('pts'))

    # Таможня
    data['Таможня'] = data['documents'].apply(lambda x: eval(x).get('custom_cleared'))

    # price
    data['price'] = data['price_info'].apply(lambda x: eval(x).get('price'))
    
    data = data[['bodyType', 'brand', 'color_hex', 'fuelType',
       'modelDate', 'name', 'numberOfDoors', 'productionDate',
       'vehicleTransmission', 'enginePower', 'mileage', 'Привод', 'Руль',
       'Состояние', 'Владельцы', 'ПТС', 'Таможня', 'price']]
    
    return(data)
# Обрабатываем датасет
df_train = preproc_train(train)
df_train.info()
# удалим строки, где отсутствует цена
df_train = df_train.dropna(axis=0, subset=['price'])

# пропуски в данных по ПТС заполним original
df_train['ПТС'] = df_train['ПТС'].fillna('ORIGINAL')

# переименуем столбец цвета
df_train = df_train.rename(columns={'color_hex': 'color'})
# Теперь поработаем с Владельцами
# Скорее всего основная часть пропусков у авто 2020 года выпуска, у коротых еще не было владельцев
df_train[df_train['productionDate']==2020].count()
def fill_driver(data):
    for row in data['productionDate']:
        if row == 2020:
            data['Владельцы'] = data['Владельцы'].fillna(0)
                
        else:
            data['Владельцы'] = data['Владельцы'].fillna(3)
    return data
df_train = fill_driver(df_train)
# Состояние, Таможню и Владельцев переведем в int
df_train['Состояние'] = df_train['Состояние'].astype('int32')
df_train['Таможня'] = df_train['Таможня'].astype('int32')
df_train['Владельцы'] = df_train['Владельцы'].astype('int32')
# Обработаем цвета
color_dict = {'CACECB': 'серебристый', 
              'FAFBFB':'белый', 
              'EE1D19':'красный', 
              '97948F':'серый',
              '660099':'пурпурный',
              '040001':'чёрный',
              '4A2197':'фиолетовый',
              '200204':'коричневый',
              '0000CC':'синий',
              '007F00':'зелёный',
              'C49648':'бежевый',
              '22A0F8':'голубой',
              'DEA522':'золотистый',
              'FFD600': 'жёлтый',
              'FF8649':'оранжевый',
              'FFC0CB':'розовый'}
df_train['color'] = df_train['color'].map(color_dict)
# Унифицируем значения в столбце bodyType
df_train['bodyType'] = df_train['bodyType'].apply(lambda x: x.split(' ')[0].lower())
df_test = pd.read_csv(DIR_TEST+'test.csv')
# Оставим только нужные колонки
df_test = df_test[['bodyType', 'brand', 'color', 'fuelType',
       'modelDate', 'name', 'numberOfDoors', 'productionDate',
       'vehicleTransmission', 'enginePower', 'mileage', 'Привод', 'Руль',
       'Состояние', 'Владельцы', 'ПТС', 'Таможня']]
df_test.head(2)
# функция для обработки тестового датафрейма
def preproc_test(data):
    data['bodyType'] = data['bodyType'].apply(lambda x: x.lower().split(' ')[0])
    
    data['enginePower'] = data['enginePower'].apply(lambda x: x.split(' ')[0])
    
    data['Владельцы'] = data['Владельцы'].apply(lambda x: int(x.split()[0]))
    
    dict_transmission = {'автоматическая':'AUTOMATIC','механическая':'MECHANICAL','роботизированная':'ROBOT',
                         'вариатор':'VARIATOR'}
    data['vehicleTransmission'] = data['vehicleTransmission'].map(dict_transmission)
    
    dict_steering_wheel = {'Левый':'LEFT','Правый':'RIGHT'}
    data['Руль'] = data['Руль'].map(dict_steering_wheel)
    
    dict_pts = {'Оригинал':'ORIGINAL','Дубликат':'DUPLICATE'}
    data['ПТС'] = data['ПТС'].map(dict_pts)
    
    dict_condition = {'Не требует ремонта':1,'Требует ремонта':0}
    data['Состояние'] = data['Состояние'].map(dict_condition)
    
    dict_customs = {'Растаможен':1,'Не растаможен':0}
    data['Таможня'] = data['Таможня'].map(dict_customs)
    
    for i in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate']:
        data[i] = data[i].astype('int32')
    return data
df_test2 = preproc_test(df_test)
import time
from datetime import datetime, timedelta
year = datetime.now().year
def feature_eng(data):
    # найдем, сколько лет нашей крошке
    data['how_old'] = data['productionDate'].apply(lambda x: year - x)

    # узнаем, как сильно ее шатали
    data['mile_per_year'] = data['mileage'] / data['how_old']
    data['mile_per_year'] = data['mile_per_year'].fillna(data['mileage'])
    
    # прикинем, на сколько свежа сама модель
    data['old_model'] = data['modelDate'].apply(lambda x: year - x)
    
    return data
df_test2 = feature_eng(df_test2)
df_train = feature_eng(df_train)
X = df_train.drop(['price'], axis=1,)
y = df_train.price.values
# Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
# Вот таким упражнением запилить актуальный список категориальных переменных не получается...
cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 3000)[0].tolist()
# ... поэтому объявим его вручную
cat_features_ids  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
#

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
model.save_model('catboost_single_model_baseline.model')
predict_submission = model.predict(df_test2)
predict_submission
sample_submission = pd.read_csv('../input/data-for-car-price-prediction-model-sub/sample_submission.csv')
sample_submission['price'] = predict_submission
sample_submission.to_csv(f'submission_v{VERSION}.csv', index=False)
sample_submission.head(10)
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
    X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]
    # model for this fold
    model = cat_model(y_train, X_train, X_test, y_test,)
    # score model on test
    test_predict = model.predict(X_test)
    test_score = mape(y_test, test_predict)
    score_ls.append(test_score)
    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")
    # submissions
    submissions[f'sub_{idx+1}'] = model.predict(df_test2)
    model.save_model(f'catboost_fold_{idx+1}.model')
    
print(f'Mean Score: {np.mean(score_ls):0.3f}')
print(f'Std Score: {np.std(score_ls):0.4f}')
print(f'Max Score: {np.max(score_ls):0.3f}')
print(f'Min Score: {np.min(score_ls):0.3f}')
