import numpy as np
import pandas as pd
import sys
import re
import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from collections import defaultdict

from scipy.stats import pearsonr

pd.set_option('display.max_rows', 50) 
pd.set_option('display.max_columns', 50)
VERSION    = 4
DIR_TRAIN  = '../input/200629-autoru/' 
DIR_TEST   = '../input/sf-dst-car-price/'
VAL_SIZE   = 0.3   # 33%
N_FOLDS    = 5

# CATBOOST
ITERATIONS = 2000
LR         = 0.1

# RANDOM_SEED
RANDOM_SEED = 42
# Функция для вывода со статисчтической иноформацией о признаке
def feature_info(_column, _bins = 0, _normalize=True, _values=False, _threshold=10):
    
    if(_bins==0):
        _bins = len(_column.value_counts(dropna=True, normalize=_normalize))
            
    print(f"Тип признака: {_column.dtype}\nЗначения признака:")
    
    
    if _column.dtype != 'object':
        _column.hist(bins = _bins, align='left')
        display(_column.describe())
        if _values == True:
            display(pd.DataFrame(_column.value_counts(dropna=False, normalize=_normalize).round(3)))
       # plt.plot(_column,pupils['score'])
    
    else:
        display(pd.DataFrame(_column.value_counts(dropna=False, normalize=_normalize).round(3)))
    
    print(f"Различных значений c учётом NaN: {len(_column.value_counts(dropna=False, normalize=_normalize))}\n\
              \t  без учёта NaN: {len(_column.value_counts(dropna=True, normalize=_normalize))} \nПропусков: {_column.isnull().sum()}")
    print(f"Значений, встретившихся в столбце более {_threshold} раз:"#Число 10 взято для ориентира, можно брать другое
      , (_column.value_counts()>_threshold).sum())
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f'RMSE = {rmse:.2f}, MAE = {mae:.2f}, R-sq = {r2:.2f}, MAPE = {mape:.2f} ')
train = pd.read_csv(DIR_TRAIN+'train.csv', sep=';', encoding ='utf-8')
train.head(5)
train.info()
test = pd.read_csv(DIR_TEST+'test.csv', sep=',', encoding ='utf-8')
test.head(5)
test.info()
# Переименование столбцов как в тренировочной выборке
test.rename(columns={'mileage': 'kmAge', 'Комплектация': 'comlpectation', 'Привод':'drive', 
                     'Руль':'whilleType', 'Состояние': 'condition', 'Владельцы':'#owners',
                    'ПТС':'pts', 'Таможня':'castoms', 'Владение':'inUse'}, inplace=True)
# Следующие признаки преобразуются к типу int
test.modelDate = test.modelDate.astype(int)
test.productionDate = test.productionDate.astype(int)
test.kmAge = test.kmAge.astype(int)

# Признак vehicleConfiguration удаляется т.к. информация из него содержится в других признаках
test.drop(['vehicleConfiguration'], axis=1, inplace=True)

# Признак description не использовался при построении модели
test.drop(['description'], axis=1, inplace=True)

# Следующий признак совпадает с индексом
#test.drop(['id'],axis=1,inplace = True)
# Предобработка признака bodyType
feature_info(test.bodyType)
test.bodyType = test.bodyType.apply(lambda x: re.sub("\d дв\.", "", x))
feature_info(test.name)
bmw = train[train.brand=='BMW']
#bmw = pd.read_csv("d:\\skillfactory\\Real Data Science\\4.Auto_pickup\\BMW_char.csv", sep=';', encoding='utf-8')
BMW = bmw.groupby(['modelDate', 'name']).count().reset_index(level=1)['name']


def process_test_name(row):
    name_parts = row['name'].split(sep=' ')
    if row['modelDate'] in BMW.keys():
        if type(BMW[row['modelDate']]) is str:
           models = [BMW[row['modelDate']]]
        else:
           models = BMW[row['modelDate']].values

        for model in models:
            if name_parts[1] == 'xDrive':
                if (name_parts[0] in model) and (name_parts[1] in model):
                    return model
            else:
                 if name_parts[0] in model:
                     return model

        else:
            models = bmw['name'].unique()
            for model in models:
                if name_parts[1] == 'xDrive':
                    if (name_parts[0] in model) and (name_parts[1] in model):
                        return model
                else:
                    if name_parts[0] in model:
                        return model

    return row['name']


test['name'] = test.apply(process_test_name, axis =1)
feature_info(test.engineDisplacement)
test.engineDisplacement = test.engineDisplacement.apply(lambda x: "1.6 LTR" if x=='undefined LTR' else x)
test.engineDisplacement = test.engineDisplacement.apply(lambda x: (re.sub(" LTR", "", x))).astype(float)
feature_info(test.enginePower)
test.enginePower = test.enginePower.apply(lambda x: (re.sub(" N12", "", x))).astype(int)
most_valueble_options = ['led_lights','tyres_contol', 'elecrtic_back', 'electic_rear_seats', 'bottom_start', 'window_airbag',
               'no_key', 'electrick_mirror', 'anti_crash', 'hsa', 'rain_control', 'leather', 'warm_wheel', 'light_control',
               'premium_audio', 'line_control', 'rear_sits_memory', 'start_stop', 'leather_wheel', 'wheel_controls',
                'navigation'] # Получен на основе корреляционного анализа с целевой переменной

def fill_components_item(value):
    if item in value:
        return 1
    else:
        return 0
    
def proc_test_complectation(x):
    if  pd.isnull(x):
        return []

    x = x.replace("['", '')
    x = x.replace("']", '')
    x = x.replace('Светодиодные фары', 'led_lights')
    x = x.replace('Датчик давления в шинах', 'tyres_contol')
    x = x.replace('Электропривод крышки багажника', 'elecrtic_back')
    x = x.replace('Электрорегулировка передних сидений', 'electic_rear_seats')
    x = x.replace('Запуск двигателя с кнопки', 'bottom_start')
    x = x.replace('Подушки безопасности оконные(шторки)', 'window_airbag')
    x = x.replace('Система доступа без ключа', 'no_key')
    x = x.replace('Электроскладывание зеркал', 'electrick_mirror')
    x = x.replace('Система предотвращения столкновения', 'anti_crash')
    x = x.replace('Система помощи при старте в гору (HSA)', 'hsa')
    x = x.replace('Датчик дождя', 'rain_control')
    x = x.replace('Кожа (Материал салона)', 'leather')
    x = x.replace('Обогрев рулевого колеса', 'warm_wheel')
    x = x.replace('Датчик света', 'light_control')
    x = x.replace('Премиальная аудиосистема', 'premium_audio')
    x = x.replace('Система контроля за полосой движения', 'line_control')
    x = x.replace('Память передних сидений', 'rear_sits_memory')
    x = x.replace('Система «старт - стоп»', 'start_stop')
    x = x.replace('Отделка кожей рулевого колеса', 'leather_wheel')
    x = x.replace('Мультифункциональное рулевое колесо', 'leather_wheel')
    x = x.replace('Подрулевые лепестки переключения передач', 'leather_wheel')
    x = x.replace('Навигационная система', 'navigation')

    values = json.loads(x)
    result = []
    for item in values:
        if 'values' in item.keys():
            result.extend(item['values'])
    return result
test['comlpectation'] = test['comlpectation'].apply(proc_test_complectation)
test['num_options'] = test['comlpectation'].apply(lambda x: len(x))
complectation_items = defaultdict(int)
for value in test['comlpectation']:
    for item in value:
         complectation_items[item] += 1

for item in most_valueble_options:
    test[item] = test['comlpectation'].apply(fill_components_item)

test.drop(['comlpectation'], axis=1, inplace = True)
test['#owners'] = test['#owners'].apply(lambda x: (re.sub("\D", "", x))).astype(int)
feature_info(test.inUse)
test.inUse.fillna('0',inplace=True)
def reshape_years(inuse):
    word = inuse.split(sep=' ')
    years = 0
    months = 0

    if len(word) >= 2:
        for i in range(0, len(word)):
            if word[i].isdigit() and word[i + 1][:3] == 'мес':
                months = int(word[i])
            elif word[i].isdigit() and (word[i + 1][:3] == 'лет' or word[i + 1][:3] == 'год'):
                years = int(word[i])
    return (years * 12 + months)   
    
test['inUse']=test.inUse.apply(reshape_years)
def fill_tax(x):
    if x in bmw.name.values:
        tax = bmw[bmw.name == x].tax.mean()
    else:
        tax = bmw.tax.mean()

    return tax

test['tax'] = test.name.apply(fill_tax)
test['total_tax'] = test.inUse*test['tax']/12
test['km_per_month'] = test['kmAge']/((2020 - test['productionDate'])*12+1)
test.head(5)
test.head(5)
train_features = train.columns.values.tolist()
test_features = test.columns.values.tolist()
for feature in train_features:
    if feature not in test_features:
        print(feature)
train.drop(['place', 'VIN', 'seller', 'exchange'], axis=1, inplace =True)
train.drop(['warm_wheel', 'elecrtic_back', 'navigation', 'wheel_controls', 'electic_rear_seats', 'led_lights',
            'hsa', 'window_airbag'], axis=1, inplace=True)
test.drop(['warm_wheel', 'elecrtic_back', 'navigation', 'wheel_controls', 'electic_rear_seats', 'led_lights',
            'hsa', 'window_airbag'], axis=1, inplace=True)
# Копии датасетов с переведёнными в числовое представление признаками
train_num = train.copy(deep=True)
test_num = test.copy(deep=True)
#name
encoder = LabelEncoder()

# name_mean =train.groupby('name').price.mean()
# mean_bmw = train[train.brand=='BMW'].price.mean()

# mean_bmw
names =train.name.values.tolist() + test.name.values.tolist()
encoder.fit(names)
train_num.name = encoder.transform(train_num.name)
test_num.name = encoder.transform(test_num.name)

# train.name = train.name.apply(lambda x: name_mean.loc[x] if x in name_mean.index else mean_bmw)
# test.name = test.name.apply(lambda x: name_mean.loc[x] if x in name_mean.index else mean_bmw)
# train_num.name = train_num.name.apply(lambda x: name_mean.loc[x] if x in name_mean.index else mean_bmw)
# test_num.name = test_num.name.apply(lambda x: name_mean.loc[x] if x in name_mean.index else mean_bmw)
#brand
# brand_mean =train.groupby('brand').price.mean()
# train.brand = train.brand.apply(lambda x: brand_mean.loc[x])
# test.brand = test.brand.apply(lambda x: brand_mean.loc[x])
# train_num.brand = train_num.brand.apply(lambda x: brand_mean.loc[x])
# test_num.brand = test_num.brand.apply(lambda x: brand_mean.loc[x])
train.brand = train.brand.astype('category')
test.brand = test.brand.astype('category')
train_num.brand = encoder.fit_transform(train_num.brand)
test_num.brand =encoder.transform(test_num.brand)
test.head(5)
train_num
# color

train_num.color = encoder.fit_transform(train_num.color)
test_num.color = encoder.transform(test_num.color)
# bodyType

train_num.bodyType = encoder.fit_transform(train_num.bodyType)
test_num.bodyType = encoder.transform(test_num.bodyType)
# fuelType

train_num.fuelType = encoder.fit_transform(train_num.fuelType)
test_num.fuelType = encoder.transform(test_num.fuelType)
# vehicleTransmission

train_num.vehicleTransmission = encoder.fit_transform(train_num.vehicleTransmission)
test_num.vehicleTransmission = encoder.transform(test_num.vehicleTransmission)
# whilleType

train_num.whilleType = encoder.fit_transform(train_num.whilleType)
test_num.whilleType = encoder.transform(test_num.whilleType)
# drive

train_num.drive = encoder.fit_transform(train_num.drive)
test_num.drive = encoder.transform(test_num.drive)
# condition

train_num.condition = encoder.fit_transform(train_num.condition)
test_num.condition = encoder.transform(test_num.condition)
# castoms

train_num.castoms = encoder.fit_transform(train_num.castoms)
test_num.castoms = encoder.transform(test_num.castoms)
# pts

train_num.pts = encoder.fit_transform(train_num.pts)
test_num.pts = encoder.transform(test_num.pts)
test = test.reindex(columns=train.drop(['price'], axis=1).columns)
corr_matrix = train.corr()
corr_features = np.abs(corr_matrix[np.abs(corr_matrix.price)>0.00].price).sort_values(ascending=False)
print("Отсортированный в порядке убывания список коррелирующих с целевой переменной признаков:\n",corr_features[1:])
for index in corr_matrix.index:
    for column in corr_matrix.columns:
        if corr_matrix[index][column]>0.7 and index!=column:
            print (index, column, corr_matrix[index][column])
test
train_num.drop(['modelDate', 'engineDisplacement', 'inUse', 'electrick_mirror', 'anti_crash', 'start_stop', 'rain_control',
                'leather', 'whilleType', 'pts', 'castoms', 'tyres_contol', 'bottom_start', 'light_control', 'no_key',
                'premium_audio', 'leather_wheel'], axis=1, inplace=True)
test_num.drop(['modelDate', 'engineDisplacement', 'inUse', 'electrick_mirror', 'anti_crash', 'start_stop', 'rain_control',
                'leather', 'whilleType', 'pts', 'castoms', 'tyres_contol', 'bottom_start', 'light_control', 'no_key',
                'premium_audio', 'leather_wheel'], axis=1, inplace=True)
train.drop(['modelDate', 'engineDisplacement', 'inUse', 'electrick_mirror', 'anti_crash', 'start_stop', 'rain_control',
                'leather', 'whilleType', 'pts', 'castoms', 'tyres_contol', 'bottom_start', 'light_control', 'no_key',
                'premium_audio', 'leather_wheel'], axis=1, inplace=True)
test.drop(['modelDate', 'engineDisplacement', 'inUse', 'electrick_mirror', 'anti_crash', 'start_stop', 'rain_control',
                'leather', 'whilleType', 'pts', 'castoms', 'tyres_contol', 'bottom_start', 'light_control', 'no_key',
                'premium_audio', 'leather_wheel'], axis=1, inplace=True)
X = train.drop(['price'], axis=1)
X_num = train_num.drop(['price'], axis=1)
y = train.price
y_num = train_num.price
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=VAL_SIZE, random_state=RANDOM_SEED)
X_num_train, X_num_test, y_num_train, y_num_test =  train_test_split(X_num, y_num, test_size=VAL_SIZE, random_state=RANDOM_SEED)
cat_features = [ 'brand','name', 'color', 'bodyType', 'fuelType', 'vehicleTransmission', 'drive',  
                'condition']# 'whilleType', 'pts',, 'castoms'
model = CatBoostRegressor(iterations = ITERATIONS, learning_rate = LR, random_seed = RANDOM_SEED,
                          eval_metric='MAPE', custom_metric=['R2', 'MAE'])
model.fit(X_train, y_train, cat_features = cat_features, eval_set=(X_test, y_test), verbose_eval = 100,
          use_best_model = True, plot = True)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh')
feat_importances.nlargest(30)
y_pred = model.predict(test)
X_num_test
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=10, random_state=RANDOM_SEED) #max_depth = 15, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 100, verbose = 100)
forest.fit(X_num_train, y_num_train)
y_pred = forest.predict(X_num_test)
print_regression_metrics(y_num_test, y_pred)
y_pred_for = forest.predict(test_num.drop(['id'], axis=1))
feat_importances = pd.Series(forest.feature_importances_, index=X.columns)
feat_importances.nlargest(30).plot(kind='barh')
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor

model_bet = BaggingRegressor(ExtraTreeRegressor(random_state=RANDOM_SEED))
model_bet.fit(X_num_train, y_num_train)
y_pred = model_bet.predict(X_num_test)
print_regression_metrics(y_num_test, y_pred)
y_pred_bag = model_bet.predict(test_num.drop(['id'], axis=1))
from sklearn.base import clone

def compute_meta_feature(clf, X_train, X_test, y_train, cv):

    X_meta_train = np.zeros_like(y_train, dtype = np.float32)
    X_meta_test = np.zeros(len(X_test), dtype=np.float32)
    for train_fold_index, predict_fold_index in cv.split(X_train):
        X_fold_train, X_fold_predict = X_train.iloc[train_fold_index], X_train.iloc[predict_fold_index]
        y_fold_train = y_train[train_fold_index]
        folded_clf = clone(clf)
        
        if type(clf).__name__ == 'CatBoostRegressor':
            folded_clf.fit(X_fold_train, y_fold_train, cat_features=cat_features_ids, verbose_eval = 100)
        else:
            folded_clf.fit(X_fold_train, y_fold_train)
            
        X_meta_train[predict_fold_index] = folded_clf.predict(X_fold_predict)
        print_regression_metrics(X_meta_train[predict_fold_index], y_train.iloc[predict_fold_index])
        X_meta_test += folded_clf.predict(X_test)
    X_meta_test = X_meta_test / cv.n_splits

    return X_meta_train, X_meta_test



def generate_meta_features(classifiers, X_train, X_test, y_train, cv):
    features = [
        compute_meta_feature(clf, X_train, X_test, y_train, cv)
        for clf in tqdm(classifiers)
    ]

    stacked_features_train = np.stack([
        features_train for features_train, features_test in features
        ],axis=-1)

    stacked_features_test = np.stack([
        features_test for features_train, features_test in features
        ],axis=-1)

    return stacked_features_train, stacked_features_test

cv = KFold(n_splits=5, shuffle=True, random_state=42)
model_rf = RandomForestRegressor(n_estimators=10, random_state=RANDOM_SEED)
model_bet = BaggingRegressor(ExtraTreeRegressor(random_state=RANDOM_SEED))
model_cb = CatBoostRegressor( 
                          #learning_rate = 0.1,
                          random_seed = 42,
                          eval_metric='MAPE')
X_num_train.reset_index(drop=True,inplace = True)
y_num_train.reset_index(drop=True, inplace = True)
stacked_features_train, stacked_features_test = generate_meta_features([model_rf,model_bet], X_num_train, 
                                                                       test_num.drop(['id'], axis=1),
                                                                       y_num_train, cv)


from sklearn.linear_model import Ridge
final_model = Ridge(alpha=20).fit(stacked_features_train, y_num_train)
y_pred = np.round((final_model.predict(stacked_features_test)/1000))*1000
#print_regression_metrics(y_test, y_pred)
y_pred_stck = np.round((final_model.predict(stacked_features_test)/1000))*1000
test['price'] =  y_pred_stck
test[['price']].to_csv(f'MBS1983_submission_stack_v{VERSION}_BMW.csv', index=True)
