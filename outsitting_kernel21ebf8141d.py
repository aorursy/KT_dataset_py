#!pip install catboost
# Импортируем необходимые библиотеки
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from tqdm.notebook import tqdm
from catboost import CatBoostRegressor
import json
import sys
import re
print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
!pip freeze > requirements.txt
RANDOM_SEED = 42

VERSION    = 5
DIR_TRAIN  = '../input/sfcarprice/' # подключил к ноутбуку свой внешний датасет
DIR_TEST   = '../input/sf-dst-car-price/'
VAL_SIZE   = 0.1   # 33%
N_FOLDS    = 5

# CATBOOST
ITERATIONS = 6000
LR         = 0.05
# обучающтй датасет
df = pd.read_csv('new_data_99_06_03_13_04.csv')
# обучающтй датасет
df_t = pd.read_csv('test.csv')
pd.options.display.max_rows = 20
pd.options.display.max_columns = len(df.columns)+5
df.head(2)
df_t.head(2)
df.info()
df_t.info()
# обучающтй датасет
df = pd.read_csv('new_data_99_06_03_13_04.csv')
# обучающтй датасет
df_t = pd.read_csv('test.csv')
# Удалим некоторые столбцы в тренировочном датасете
df.drop(['Unnamed: 0'], inplace = True, axis = 1)
df.drop(['Владение'], inplace = True, axis = 1)
#df.drop(['name'], inplace = True, axis = 1)
#df.drop(['description'], inplace = True, axis = 1)

# Пропусков у нас мало, поэтому просто удалим строки с пропусками
df.dropna(inplace = True)
list(df)
# переведем некоторые названия колонок на английский язык
df.columns = ['bodyType',
 'brand',
 'color',
 'fuelType',
 'modelDate',
 'name',
 'numberOfDoors',
 'productionDate',
 'vehicleTransmition',
 'engineDisplacement',
 'enginePower',
 'description',
 'mileage',
 'configuration',
 'driveType',
 'steeringWheel',
 'owners',
 'technicalPassport',
 'customs',
 'Price']
df.head(2)
df_t.head(2)
df.info()
# Удалим некоторые столбцы в тестовом датасете
df_t.drop(['id'], inplace = True, axis = 1)
df_t.drop(['Владение'], inplace = True, axis = 1)
df_t.drop(['Состояние'], inplace = True, axis = 1)
#df_t.drop(['name'], inplace = True, axis = 1)
df_t.drop(['vehicleTransmission'], inplace = True, axis = 1)
#df_t.drop(['description'], inplace = True, axis = 1)

# Пропусков у нас тожн немного, поэтому просто удалим строки с пропусками
df_t.dropna(inplace = True)
list(df_t)
df_t.columns = ['bodyType',
 'brand',
 'color',
 'fuelType',
 'modelDate',
 'name',
 'numberOfDoors',
 'productionDate',
 'vehicleTransmition',
 'engineDisplacement',
 'enginePower',
 'description',
 'mileage',
 'configuration',
 'driveType',
 'steeringWheel',
 'owners',
 'technicalPassport',
 'customs']
df_t.bodyType.value_counts()
df.bodyType.value_counts()
# 10 самых популярных типов кузова составляют 94.76% данных
(df.bodyType.value_counts()[:10].sum() / len(df) ) * 100
body = {}
b = 1
for a in df.bodyType.value_counts().index:
    body[a.lower()] = b
    b += 1
#df['bodyType'] = df['bodyType'].map(body)
df.bodyType = df.bodyType.apply(lambda x: int(body[str(x).lower()]))
df.bodyType = df.bodyType.apply(lambda x: 11 if x>=11 else x)
df.bodyType.value_counts()
df_t.bodyType = df_t.bodyType.apply(lambda x: int(body[str(x).lower()]))
df_t.bodyType = df_t.bodyType.apply(lambda x: 11 if x>=11 else x)
df_t.bodyType.value_counts()
df.brand.unique()
df_t.brand.unique()
brand = {}
c = 1
for b in df.brand.unique():
    brand[b.upper()] = c
    c += 1
df.brand = df.brand.apply(lambda x: int(brand[str(x).upper()]))
df_t.brand = df_t.brand.apply(lambda x: int(brand[str(x).upper()]))
df.color.unique()
df_t.color.unique()
color = {}
i = 1
for c in df.color.unique():
    color[c] = i
    i += 1
color
color_t = {'чёрный': 1,
 'белый': 2,  #светло-серый
 'серый': 3,  #≈серый
 'синий': 4,  #≈тёмно-синий
 'фиолетовый': 4, #≈тёмный фиолетово-синий
 'красный': 6,
 'серебристый': 7, #≈бледный серо-циановый
 'зелёный': 8,
 'голубой': 9,
 'фиолетовый': 10, # Тёмный пурпурно-фиолетовый 
 'жёлтый': 11,
 'коричневый': 12,
 'бежевый': 13,
 'оранжевый': 14,
 'золотистый': 15,
 'пурпурный': 16} #розовый
df.color = df.color.apply(lambda x: int(color[str(x).upper()]))
df_t.color = df_t.color.apply(lambda x: int(color_t[str(x).lower()]))
df.fuelType.unique()
df_t.fuelType.unique()
gas = {}
i = 1
for c in df.fuelType.unique():
    gas[c] = i
    i += 1
gas
df.fuelType = df.fuelType.apply(lambda x: int(gas[str(x).lower()]))
df_t.fuelType = df_t.fuelType.apply(lambda x: int(gas[str(x).lower()]))
df.modelDate.unique()
df.modelDate = 2021 - df.modelDate
df_t.modelDate.unique()
df_t.modelDate = 2021 - df_t.modelDate
df.numberOfDoors.value_counts()
df_t.numberOfDoors.value_counts()
df.numberOfDoors.replace(0, 4, inplace=True)
df.productionDate.unique()
df.productionDate = 2021 - df.productionDate
df_t.productionDate.unique()
df_t.productionDate = 2021 - df_t.productionDate
df.vehicleTransmition.value_counts()
df_t.vehicleTransmition.value_counts()
transmition = {}
i = 1
for t in df.vehicleTransmition.unique():
    transmition[t] = i
    i += 1
transmition
df.vehicleTransmition = df.vehicleTransmition.apply(lambda x: int(transmition[str(x).upper()]))
df_t.vehicleTransmition = df_t.vehicleTransmition.apply(lambda x: 2 if 'AUTOMATIC' in x
                                                           else 3 if 'ROBOT' in x
                                                           else 4 if 'VARIATOR' in x
                                                           else 1)
df.name.value_counts()
engine = {}
f = re.compile(r'\d+\.\d+')
string = []
for i in df.name.value_counts().index:
    s = f.findall(i)
    if len(s) == 0:
        s = [0.0]
    engine[i] = int(float(s[0])*1000)
df.tail()
df.engineDisplacement = df.name.apply(lambda x: engine[x])
df_t.engineDisplacement.unique()
df_t.engineDisplacement = df_t.engineDisplacement.apply(lambda x: -1 if x == 'undefined LTR'
                                                        else int(float(str(x).replace(' LTR', ''))*1000) )
df.enginePower.value_counts()
df_t.enginePower.value_counts()
df_t.enginePower = df_t.enginePower.apply(lambda x: int(x.replace(' N12', '')))
df_t.info()
df.mileage.value_counts()
df_t.mileage.value_counts()
df.configuration.value_counts().head(6)
def conf_pars(s):
    if pd.isnull(s):
        return []
    #уберем пробелы и кавычки
    s = s.replace(' ', '')
    s = s.replace("'", "")
    start = s.find('[')
    finish = s.find(']')
    if start == -1 or finish == -1:
        return []
    s = s[start+1:finish]
    return s.split(',')
df.configuration = df.configuration.apply(conf_pars)
import collections
cc = collections.Counter()
for i in df.configuration:
    if i != []:
        for j in i:
            cc[j] += 1
len(cc)
# всего 155 опций
cc.most_common(15)
df_t.configuration.value_counts().head(2)
def conf_pars_t(s):
    if s != s or pd.isnull(s):
        return []
    # очистим строку от мусора
    s = s.replace("['",'')
    s = s.replace("']",'')
    
    elements = json.loads(s)
    res = []
    for item in elements:
        if 'values' in item.keys():
            res.extend(item['values'])
    return res
df_t.configuration = df_t.configuration.apply(conf_pars_t)
cc_t = collections.Counter()
for i in df_t.configuration:
    if i != []:
        for j in i:
            cc_t[j] += 1
cc.most_common(20)
conf_replace = {'Антиблокировочная система (ABS)': 'abs',
 'Центральный замок': 'lock',
 'Бортовой компьютер': 'computer',
 'Система стабилизации (ESP)': 'esp',
 'Электропривод зеркал': 'electro-mirrors',
 'Противотуманные фары': 3011,
 'Подогрев передних сидений': 'front-seats-heat',
 'Подушка безопасности водителя': 'airbag-driver',
 'Электростеклоподъёмники передние': 'electro-window-front',
 'Подушка безопасности пассажира': 'airbag-passenger',
 'Подушки безопасности боковые': 'airbag-side',
 'Электростеклоподъёмники задние': 2842,
 'Иммобилайзер': 'immo',
 'Электрообогрев боковых зеркал': 2758,
 'Отделка кожей рулевого колеса': 2686,
 'Мультифункциональное рулевое колесо': 2662,
 'AUX': 2604,
 'Датчик дождя': 2563,
 'Передний центральный подлокотник': 2515,
 'Датчик света': 2447,
 'Омыватель фар': 2352,
 'Кожа (Материал салона)': 2283,
 'Автоматический корректор фар': 2270,
 'Парктроник задний': 2228,
 'Подушки безопасности оконные (шторки)': 2210,
 'USB': 2145,
 'Регулировка руля по вылету': 'wheel-configuration2',
 'Bluetooth': 2124,
 'Запуск двигателя с кнопки': 2116,
 'Обогрев рулевого колеса': 2081,
 'Крепление детского кресла (задний ряд) ISOFIX': 2076,
 'Розетка 12V': 1998,
 'Регулировка руля по высоте': 'wheel-configuration1',
 'Ксеноновые/Биксеноновые фары': 1923,
 'Круиз-контроль': 1895,
 'Электрорегулировка передних сидений': 1855,
 'Прикуриватель и пепельница': 1826,
 'Регулировка передних сидений по высоте': 1809,
 'Парктроник передний': 1734,
 'Система «старт-стоп»': 1688,
 'Складывающееся заднее сиденье': 1681,
 'Тонированные стекла': 1637,
 'Датчик давления в шинах': 1633,
 'Аудиосистема': 1616,
 'Третий задний подголовник': 1610,
 'Легкосплавные диски': 1595,
 'Активный усилитель руля': 1573,
 'Система помощи при старте в гору (HSA)': 1551,
 'Навигационная система': 1546,
 'Электроскладывание зеркал': 1520,
 'Усилитель руля': 1473,
 'Климат-контроль 2-зонный': 1464,
 'Тёмный салон': 1434,
 'Антипробуксовочная система (ASR)': 1382,
 'Электропривод крышки багажника': 1346,
 'Сигнализация': 1139,
 'Отделка кожей рычага КПП': 1134,
 'Блокировка замков задних дверей': 1106,
 'Климат-контроль 1-зонный': 1019,
 'Количество мест: 5': 1004}
df.driveType.unique()
df_t.driveType.unique()
drive = {'передний': 2, 'полный': 1, 'задний': 3}
df.driveType = df.driveType.apply(lambda x: drive[x])
df_t.driveType = df_t.driveType.apply(lambda x: drive[x])
df.steeringWheel.value_counts()
df_t.steeringWheel.value_counts()
sw = {'LEFT': 1, 'RIGHT': 2}
df.steeringWheel = df.steeringWheel.apply(lambda x: sw[x])
sw_t = {'Левый': 1, 'Правый': 2}
df_t.steeringWheel = df_t.steeringWheel.apply(lambda x: sw_t[x])
df.owners.value_counts()
df_t.owners.value_counts()
own = {'3 или более': 3, '2\xa0владельца': 2, '1\xa0владелец': 1}
df_t.owners = df_t.owners.apply(lambda x: own[x])
df_t
df.technicalPassport.value_counts()
df_t.technicalPassport.value_counts()
tp = {'ORIGINAL': 1, 'DUPLICATE': 2}
df.technicalPassport = df.technicalPassport.apply(lambda x: tp[x])
tp_t = {'Оригинал': 1, 'Дубликат': 2}
df_t.technicalPassport = df_t.technicalPassport.apply(lambda x: tp_t[x])
df.customs.unique()
df_t.customs.unique()
df.customs = 1
df_t.customs = 1
df.head(2)
df_t.head(2)
l1 = df_t.name.value_counts().index
l0 = df.name.value_counts().index
for i in l1:
    s = str(' '.join(i.split(' ')[1:]))
    if s not in l0:
        print(s, i)
df['tax'] = df.enginePower.apply(lambda x: x*12 if x<=100 
                                           else x*25 if x>100 and x<=125
                                           else x*35 if x>125 and x<=150
                                           else x*45 if x>150 and x<=175
                                           else x*55 if x>175 and x<=200
                                           else x*65 if x>200 and x<=225
                                           else x*75 if x>225 and x<=250
                                           else x*150)
df['tax_cat'] = df.enginePower.apply(lambda x: 1 if x<=100 
                                           else 2 if x>100 and x<=125
                                           else 3 if x>125 and x<=150
                                           else 4 if x>150 and x<=175
                                           else 5 if x>175 and x<=200
                                           else 6 if x>200 and x<=225
                                           else 7 if x>225 and x<=250
                                           else 8)
df_t['tax'] = df_t.enginePower.apply(lambda x: x*12 if x<=100 
                                           else x*25 if x>100 and x<=125
                                           else x*35 if x>125 and x<=150
                                           else x*45 if x>150 and x<=175
                                           else x*55 if x>175 and x<=200
                                           else x*65 if x>200 and x<=225
                                           else x*75 if x>225 and x<=250
                                           else x*150)
df_t['tax_cat'] = df_t.enginePower.apply(lambda x: 1 if x<=100 
                                           else 2 if x>100 and x<=125
                                           else 3 if x>125 and x<=150
                                           else 4 if x>150 and x<=175
                                           else 5 if x>175 and x<=200
                                           else 6 if x>200 and x<=225
                                           else 7 if x>225 and x<=250
                                           else 8)
# df.tax = df.apply(lambda x: 0 if x.engineDisplacement == 0 else x.tax)
# df.tax_cat = df.apply(lambda x: 0 if x.engineDisplacement == 0 else x.tax_cat)
# df_t.tax = df_t.apply(lambda x: 0 if x.engineDisplacement == 0 else x.tax)
# df_t.tax_cat = df_t.apply(lambda x: 0 if x.engineDisplacement == 0 else x.tax_cat)
#cols = df.columns
cols = ['bodyType', 'brand', 'color', 'fuelType', 'modelDate',
       'numberOfDoors', 'productionDate', 'vehicleTransmition',
       'engineDisplacement', 'enginePower', 'mileage',
       'driveType', 'steeringWheel', 'owners',
       'technicalPassport', 'tax', 'tax_cat']
X = df[cols]
y = df['Price']
X = X.astype(int)
y = y.astype(int)
X_val = df_t[cols]
X_val = X_val.astype(int)
X.head()
cat_features_ids = ['bodyType', 'brand', 'color', 'fuelType',
       'numberOfDoors', 'vehicleTransmition',
       'engineDisplacement', 'enginePower',
       'driveType', 'steeringWheel', 'owners',
       'technicalPassport', 'tax_cat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
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
predict_submission = model.predict(X_val)
predict_submission
sample_submission = pd.DataFrame()
sample_submission['price'] = np.ceil(predict_submission / 10000) * 10000
sample_submission['id'] = sample_submission.index
sample_submission = sample_submission[['id', 'price']]
sample_submission.to_csv(f'submission_v{VERSION}.csv', index=False)
sample_submission.tail(100)
# answer['price'] = np.ceil(answer['price'].values / 10000) * 10000 # округлили до 10 тыс
# answer = answer[['id', 'price']]
# answer.to_csv(f'submission_v{V}.csv', index=False)
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
submissions = pd.DataFrame(0,columns=["sub_1"], index=X_val.index) # куда пишем предикты по каждой модели
score_ls = []
splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, y))
for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total=N_FOLDS,):
    # use the indexes to extract the folds in the train and validation data
    X_train, y_train, X_test, y_test = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]
    model = cat_model(y_train, X_train, X_test, y_test,)
    # score model on test
    test_predict = model.predict(X_test)
    test_score = mape(y_test, test_predict)
    score_ls.append(test_score)
    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")
    # submissions
    submissions[f'sub_{idx+1}'] = model.predict(X_val)
    model.save_model(f'catboost_fold_{idx+1}.model')
    
print(f'Mean Score: {np.mean(score_ls):0.3f}')
print(f'Std Score: {np.std(score_ls):0.4f}')
print(f'Max Score: {np.max(score_ls):0.3f}')
print(f'Min Score: {np.min(score_ls):0.3f}')
submissions
V = 44
answer = pd.DataFrame()
answer['price'] = submissions.sum(axis=1) / len(submissions.columns)
answer['id'] = answer.index
answer['price'] = np.ceil(answer['price'].values / 10000) * 10000 # округлили до 10 тыс
answer = answer[['id', 'price']]
answer.to_csv(f'submission_v{V}.csv', index=False)
answer.head(10)