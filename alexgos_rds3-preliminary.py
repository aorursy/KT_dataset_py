import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
from catboost import CatBoostRegressor
import re
import requests, json
from pprint import pprint
# Фиксируем RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42
url8="https://auto.ru/-/ajax/desktop/listing/" #ссылка, куда уходит запрос

headers8='''Host: auto.ru
User-Agent: Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:75.0) Gecko/20100101 Firefox/75.0
Accept: */*
Accept-Language: ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3
Accept-Encoding: gzip, deflate, br
Referer: https://auto.ru/rossiya/cars/bmw/all/?output_type=list&page=2
x-client-app-version: 202007.24.093900
x-page-request-id: 51a84fcec5f17657e80999224cb1e87b
x-client-date: 1595679883617
x-csrf-token: 2b298e49b7f281a3ad57b2c7ce76df87df846bf7dcc24586
x-requested-with: fetch
content-type: application/json
Origin: https://auto.ru
Content-Length: 114
DNT: 1
Connection: keep-alive
Cookie: _csrf_token=2b298e49b7f281a3ad57b2c7ce76df87df846bf7dcc24586; autoru_sid=a%3Ag5f1c220c269gd64paqva86ie1em6uq5.0eb4c178b32ff7b9b020616b1b0cfdba%7C1595679244560.604800.b-yD1aM7Z3sdp6bNYTolyA.--QpizT2ujDaS901FIKAcViHCvoUpBHiwywsfBEt2ms; autoruuid=g5f1c220c269gd64paqva86ie1em6uq5.0eb4c178b32ff7b9b020616b1b0cfdba; suid=a038631120f6bb35581286086d993aea.6609dbe0de2cc274bc57028fe9d9caf1; from_lifetime=1595679872394; from=direct; yuidcs=1; X-Vertis-DC=sas; yuidlt=1; yandexuid=9155181331595679247; crookie=dAt44fOhTnZuvlk3/FX2yXRXX8Q8Tt3J3Gzu3Hy6RleWiQNGMnm2xXdfIiSNXhf8GRbd5t+F2l8F1mH0oqwRvVNkvYA=; cmtchd=MTU5NTY3OTI2NDk3MA==; bltsr=1'''.strip().split('\n')

dict_headers8={}

for header in headers8:
    key, value=header.split(': ')
    dict_headers8[key]=value
offers8=[] # создаем список 
for x in range(0,100):
    params8={
        "catalog_filter":[{"mark":"BMW"}],
        "category":"cars",
        "geo_id":[225],
        "output_type":"list",
        "page":x,
        "section":"all"
    }
    response8=requests.post(url8, json=params8, headers=dict_headers8) #делаем запрос
    data8=response8.json()
    offers8.extend(data8['offers']) 
    print("current page:",x) #визуализация постраничной загрузки
    
#записываем итерации постранично в файл
with open("car_price8.json","w") as f:
    json.dump(offers8, f)
  
#открываем файл json в переменную
with open("car_price8.json","r",encoding="utf8") as f: 
    car_price8=json.load(f)
print ("car_price8:",len(car_price8)) #смотрим сколько элементов в переменной    
  
    
#создаем список price 
price=[]
for offer in car_price8:
    if 'price' in offer['price_info']:
        price.append(offer['price_info']['price'])
    else:
        price.append(None)
print("price:",len(price)) #смотрим сколько элементов в списке  

#создаем список body_type    
bodyType=[]
for offer in car_price8:
    if 'human_name' in offer['vehicle_info']['configuration']:
        bodyType.append(offer['vehicle_info']['configuration']['human_name'])
    else:
        bodyType.append(None) 
print ("bodyType:",len(bodyType)) #смотрим сколько элементов в переменной     
        
#создаем список brand (BMW)
brand=[]
for offer in car_price8:
    if 'code' in offer['vehicle_info']['mark_info']:
        brand.append(offer['vehicle_info']['mark_info']['code'])
    else:
        brand.append(None)
print ("brand:",len(brand)) #смотрим сколько элементов в переменной 

#создаем список color(чёрный)
color=[]
for offer in car_price8:
    if 'color_hex' in offer:
        color.append(offer['color_hex'])
    else:
        color.append(None)
print ("Количество color:",len(color)) #смотрим сколько элементов в переменной

#создаем список fuelType (дизель)
fuelType=[]
for offer in car_price8:
    if 'engine_type' in offer['vehicle_info']['tech_param']:
        fuelType.append(offer['vehicle_info']['tech_param']['engine_type'])
    else:
        fuelType.append(None)
print ("fuelType:",len(fuelType)) #смотрим сколько элементов в переменной 

#создаем список modelDate(2016.0)
modelDate=[]
for offer in car_price8:
    if 'year_from' in offer['vehicle_info']['super_gen']:
        modelDate.append(offer['vehicle_info']['super_gen']['year_from'])
    else:
        modelDate.append(None)
print ("modelDate:",len(modelDate)) #смотрим сколько элементов в переменной 

#создаем список name(520d 2.0d AT (190 л.с.))
name=[]
for offer in car_price8:
    if 'human_name' in offer['vehicle_info']['tech_param']:
        name.append(offer['vehicle_info']['tech_param']['human_name'])
    else:
        name.append(None)
print ("name:",len(name)) #смотрим сколько элементов в переменной 

#создаем список numberOfDoors(4.0)
numberOfDoors=[]
for offer in car_price8:
    if 'doors_count' in offer['vehicle_info']['configuration']:
        numberOfDoors.append(offer['vehicle_info']['configuration']['doors_count'])
    else:
        numberOfDoors.append(None)
print ("numberOfDoors:",len(numberOfDoors)) #смотрим сколько элементов в переменной 

#создаем список vehicleConfiguration(SEDAN AUTOMATIC 2.0)
vehicleConfiguration=[]
for offer in car_price8:
    if 'body_type' in offer['vehicle_info']['configuration']:
        vehicleConfiguration.append(offer['vehicle_info']['configuration']['body_type'])
    else:
        vehicleConfiguration.append(None)
print ("vehicleConfiguration:",len(vehicleConfiguration)) #смотрим сколько элементов в переменной 
#дублирует предыдущий пункт body_type

#создаем список vehicleTransmission(автоматическая)
vehicleTransmission=[]
for offer in car_price8:
    if 'transmission' in offer['vehicle_info']['tech_param']:
        vehicleTransmission.append(offer['vehicle_info']['tech_param']['transmission'])
    else:
        vehicleTransmission.append(None)
print ("vehicleTransmission:",len(vehicleTransmission)) #смотрим сколько элементов в переменной 

#создаем список engineDisplacement(2.0 LTR)
engineDisplacement=[]
for offer in car_price8:
    if 'power_kvt' in offer['vehicle_info']['tech_param']:
        engineDisplacement.append(offer['vehicle_info']['tech_param']['power_kvt']/100)
    else:
        engineDisplacement.append(None)
print ("engineDisplacement:",len(engineDisplacement)) #смотрим сколько элементов в переменной 

#создаем список enginePower(190 N12)
enginePower=[]
for offer in car_price8:
    if 'power' in offer['vehicle_info']['tech_param']:
        enginePower.append(offer['vehicle_info']['tech_param']['power'])
    else:
        enginePower.append(None)
print ("enginePower:",len(enginePower)) #смотрим сколько элементов в переменной 

#создаем список description(В РОЛЬФ Ясенево представлено более 500 автомоб)
description=[]
for offer in car_price8:
    if 'description' in offer:
        description.append(offer['description'])
    else:
        description.append(None)
print ("Количество description:",len(description)) #смотрим сколько элементов в переменной 

#создаем список mileage (158836.0)
mileage=[]
for offer in car_price8:
    if 'mileage' in offer['state']:
        mileage.append(offer['state']['mileage'])
    else:
        mileage.append(None)
print ("mileage:",len(mileage)) #смотрим сколько элементов в переменной

#создаем список Комплектация(['[{"name":"Безопасность","values":["Антипробу... )
Комплектация=[]
for offer in car_price8:
    if 'available_options' in (offer['vehicle_info']['complectation']):
        Комплектация.append(offer['vehicle_info']['complectation']['available_options'])
    else:
        Комплектация.append(None)
print ("Количество Комплектация:",len(Комплектация)) #смотрим сколько элементов в переменной

#создаем список Привод(задний)
Привод=[]
for offer in car_price8:
    if 'gear_type' in offer['vehicle_info']['tech_param']:
        Привод.append(offer['vehicle_info']['tech_param']['gear_type'])
    else:
        Привод.append(None)
print ("Привод:",len(Привод)) #смотрим сколько элементов в переменной

#создаем список Руль(Левый)
Руль=[]
for offer in car_price8:
    if 'steering_wheel' in offer['vehicle_info']:
        Руль.append(offer['vehicle_info']['steering_wheel'])
    else:
        Руль.append(None)
print ("Руль:",len(Руль)) #смотрим сколько элементов в переменной

#создаем список Владельцы (1 владелец)
Владельцы=[]
for offer in car_price8:
    if 'owners_number' in offer['documents']:        
        Владельцы.append(offer['documents']['owners_number'])        
    else:
        Владельцы.append(None)
print ("Владельцы:",len(Владельцы)) #смотрим сколько элементов в переменной

#создаем список ПТС(Оригинал)
ПТС=[]
for offer in car_price8:
    if 'pts_original' in offer['documents']:        
        ПТС.append(offer['documents']['pts_original'])        
    else:
        ПТС.append(None)
print ("ПТС:",len(ПТС)) #смотрим сколько элементов в переменной

#создаем список Таможня(Растаможен)
Таможня=[]
for offer in car_price8:
    if 'custom_cleared' in offer['documents']:        
        Таможня.append(offer['documents']['custom_cleared'])        
    else:
        Таможня.append(None)
print ("Таможня:",len(Таможня)) #смотрим сколько элементов в переменной

#создаем список Владение(7 лет и 2 месяца)
Владение=[]
for offer in car_price8:
    if 'purchase_date' in offer['documents']:        
        Владение.append(offer['documents']['purchase_date'])        
    else:
        Владение.append(None)
print ("Количество Владение:",len(Владение)) #смотрим сколько элементов в переменной

df = pd.DataFrame(price)#,header=None)
df.columns=['price']
df['bodyType']=bodyType
df['brand']=brand
df['color']=color
df['fuelType']=fuelType
df['name']=name
df['productionDate']=modelDate
df['vehicleConfiguration']=vehicleConfiguration
df['vehicleTransmission']=vehicleTransmission
df['engineDisplacement']=engineDisplacement
df['enginePower']=enginePower
df['description']=description
df['mileage']=mileage
df['Комплектация']=Комплектация
df['Привод']=Привод
df['Руль']=Руль
df['Владельцы']=Владельцы
df['ПТС']=ПТС
df['Таможня']=Таможня
df['Владение']=Владение

df.info()
df.head(3)
#переведем обозначение цвета машины из Hex в "человеческий" вид
df['color']=df.apply(lambda x: 'черный' if x['color'] == '040001' else ('белый' if x['color'] == 'FAFBFB' else ('коричневый' if x['color'] == '200204' else('синий' if x['color'] == '0000CC' else ('красный' if x['color'] == 'EE1D19' else ('серебристый' if x['color'] == 'CACECB' else ('желтый' if x['color'] == 'C49648' else ('серый' if x['color'] == '97948F' else "None"))))))), axis = 1)
df['color'].value_counts()

#сделаем названия в столбце с маленькой буквы
df['bodyType']=df.apply(lambda x: x['bodyType'].lower(), axis = 1)
df['bodyType'].value_counts()

#переведем обозначение типа топлива машины для соответствия с тестовой версией
df['fuelType']=df.apply(lambda x: 'бензин' if x['fuelType'] == 'GASOLINE' else ('дизель' if x['fuelType'] == 'DIESEL' else "None"), axis = 1)
df.fuelType.value_counts()

#переведем обозначение типа привода машины для соответствия с тестовой версией
df['Привод']=df.apply(lambda x: 'полный' if x['Привод'] == 'ALL_WHEEL_DRIVE' else ('задний' if x['Привод'] == 'REAR_DRIVE'  else ('передний' if x['Привод'] == 'FORWARD_CONTROL' else "None")), axis = 1)
df.Привод.value_counts()

#переведем обозначение типа руля машины для соответствия с тестовой версией
df['Руль']=df.apply(lambda x: 'Левый' if x['Руль'] == 'LEFT' else ('Правый' if x['Руль'] == 'RIGHT' else "None"), axis = 1)
df.Руль.value_counts()

#переведем обозначение таможенного статуса машины для соответствия с тестовой версией
df['Таможня']=df.apply(lambda x: 'Растаможен' if x['Таможня'] == True else "Нерастаможен", axis = 1)
df.Таможня.value_counts()

#переведем обозначение коробки передач машины для соответствия с тестовой версией
df['vehicleTransmission']=df.apply(lambda x: 'автоматическая' if x['vehicleTransmission'] == 'AUTOMATIC' else ('механическая' if x['vehicleTransmission'] == 'MECHANICAL' else ('роботизированная' if x['vehicleTransmission'] == 'ROBOT' else "None")), axis = 1)
df.vehicleTransmission.value_counts()

df.info()
df.head(3)
df_1=df.drop(['Владение','Комплектация','description'],axis=1)
df_1.info()
df_1.head(3)
df_drop=df_1.dropna() #удаление пропущеных значений
df_drop.info()
df_drop.head(3)
#выделим из столбца 'name' литраж двигателя
df_drop['engineDisplacement']=df_drop.apply(lambda x: re.findall(r'\d\.\d', x['name']),axis=1)
df_drop['engineDisplacement']=df_drop['engineDisplacement'].str.get(0) #уберем квадратные скобки из результата
df_drop['engineDisplacement'].value_counts()
# Переводим признаки из float в int 
for feature in ['Владельцы']:
    df_drop[feature]=df_drop[feature].astype('int32')
print(df_drop.Владельцы.value_counts())

#приведем обозначение количества владельцев машины для соответствия с тестовой версией
df_drop['Владельцы']=df_drop.apply(lambda x: str(x['Владельцы'])+' владелец' if x['Владельцы']==1 else (str(x['Владельцы'])+' владельца' if 1<x['Владельцы']<5 else (str(x['Владельцы'])+' владельцев' if x['Владельцы']>4 else "None")), axis = 1)
print(df_drop.Владельцы.value_counts())

#Setup
VERSION    = 11
VAL_SIZE   = 0.33   # 33%
N_FOLDS    = 5

# CATBOOST
ITERATIONS = 2000
LR         = 0.1

#Data Preprocessing
def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
    # ################### Предобработка ############################################################## 
    # убираем не нужные для модели признаки
    df_output.drop(['Таможня', 'Состояние', 'id','ПТС'], axis=1, inplace=True,)
    
    
    # ################### fix ############################################################## 
    # Переводим признаки из float в int (иначе catboost выдает ошибку)
    for feature in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate']:
        df_output[feature]=df_output[feature].astype('int32')
    
    
    # ################### Feature Engineering ####################################################
    # тут ваш код на генерацию новых фитчей
    # ....
    
    
    # ################### Clean #################################################### 
    # убираем признаки которые еще не успели обработать, 
    df_output.drop(['Комплектация', 'description', 'Владение'], axis=1, inplace=True,)
    
    return df_output
train = df_drop  # мой подготовленный датасет для обучения модели
test = pd.read_csv('test_car.csv') # тестовый набор признаков
sample_submission = pd.read_csv('sample_submission_car.csv') #целевая тестовая переменная

train_preproc = train # обработанный подготовленный датасет 
X_sub = preproc_data(test) # обработанная тестовая выборка  

X = train_preproc.drop(['price'], axis=1,) #тренировочный набор признаков
y = train_preproc.price.values #целевая переменная датасета

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)

# список признаков
X_train.nunique()
# Keep list of all categorical features in dataset to specify this for CatBoost
cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 3000)[0].tolist()
cat_features_ids 
#Fit
model = CatBoostRegressor(iterations = ITERATIONS,
                          learning_rate = LR,
                          random_seed = RANDOM_SEED,
                          eval_metric='MAPE',
                          custom_metric=['R2', 'MAE']
                         )
model.fit(X_train, y_train, # обучение
         cat_features=cat_features_ids,
         eval_set=(X_test, y_test),
         verbose_eval=100,
         use_best_model=True,
         plot=True
         )

model.save_model('catboost_single_model_baseline.model')
#Submission
predict_submission = model.predict(X_sub) # # передача тестовой выборки в модель
print(len(predict_submission)) # вывод количества значений
print(predict_submission)# вывод результата "предсказания"
sample_submission.head(3) #фактические значения за анализируемый период
sample_submission['price1'] = predict_submission #значения прогнозной модели за анализируемый период
sample_submission['price2']=abs(sample_submission['price']-sample_submission['price1'])/sample_submission['price'] #абсолютная ошибка в процентах
sample_submission.to_csv(f'sample_submission_car{VERSION}.csv', index=False)
sample_submission.head(3)
#средняя абсолютная ошибка в процентах
MAPE=sum(sample_submission['price2'])/len(sample_submission['price2'])
MAPE
#CV смотрим вариант на кросс-валидации

def cat_model(y_train, X_train, X_test, y_test):
    model9 = CatBoostRegressor(iterations = ITERATIONS,
                              learning_rate = LR,
                              eval_metric='MAPE',
                              random_seed = RANDOM_SEED,)
    model9.fit(X_train, y_train,
              cat_features=cat_features_ids,
              eval_set=(X_test, y_test),
              verbose=False,
              use_best_model=True,
              plot=False)
    
    return(model9)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))
submissions9 = pd.DataFrame(0,columns=["sub_1"], index=sample_submission.index) # куда пишем предикты по каждой модели
score_ls9 = []
splits9 = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, y))

for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total=N_FOLDS,):
    # use the indexes to extract the folds in the train and validation data
    X_train9, y_train9, X_test9, y_test9 = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]
    # model for this fold
    model9 = cat_model(y_train9, X_train9, X_test9, y_test9,)
    # score model on test
    test_predict9 = model9.predict(X_test9)
    test_score9 = mape(y_test9, test_predict9)
    score_ls.append(test_score9)
    print(f"{idx+1} Fold Test MAPE: {mape(y_test9, test_predict9):0.3f}")
    # submissions
    submissions9[f'sub_{idx+1}'] = model9.predict(X_sub)
    model9.save_model(f'catboost_fold_{idx+1}.model')
    
print(f'Mean Score: {np.mean(score_ls):0.3f}')
print(f'Std Score: {np.std(score_ls):0.4f}')
print(f'Max Score: {np.max(score_ls):0.3f}')
print(f'Min Score: {np.min(score_ls):0.3f}')
submissions9.head(3)
#Submissions blend 

submissions9['blend'] = (submissions9.sum(axis=1))/len(submissions9.columns)
submissions9.head(3)
sample_submission['price9'] = submissions9['blend'].values
sample_submission.to_csv(f'submission_blend_v{VERSION}.csv', index=False)
sample_submission.head(3)
sample_submission['price_target1']=abs(sample_submission['price']-sample_submission['price9'])/sample_submission['price'] #абсолютная ошибка в процентах

sample_submission.head(3)
MAPE1=sum(sample_submission['price_target1'])/len(sample_submission['price_target1'])
MAPE1
