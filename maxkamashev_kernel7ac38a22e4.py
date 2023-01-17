VERSION = 33
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

is_debug = False
if not is_debug:
    # train_dataset_url="https://drive.google.com/u/0/uc?id=1HFV1106xXhrnNt5wG1nXl0X-b5Jql2Md&export=download"
    train_dataset_url="https://drive.google.com/u/0/uc?id=1MaX59-keo_h4TEGKwW-HO7Ehh_O2wVwu&export=download"
    train_orig = pd.read_csv(train_dataset_url, low_memory = False)
train = train_orig.copy()
train.info()
if not is_debug:
    test_dataset_url="https://drive.google.com/u/0/uc?id=18dDPo6GF5VSU2MaIvOk6PFjUnfBi3A42&export=download"
    test_orig = pd.read_csv(test_dataset_url)
test = test_orig.copy()
test.info()
common_cols = set()
def describe(df, field_name):
    print(f"Колонка \"{field_name}\":")
    print("------")
    print("na:", df[field_name].isna().sum())
    print("уникальных значений:", len(df[field_name].unique()))
    print("------")
    print(df[field_name].value_counts())  
df = test
field_name = 'bodyType'
describe(df, field_name)
def extract(x):
    ss = str(x).split()
    if len(ss) > 2:
        return float(ss[1])
    else:
        return None
df['дв'] = df[field_name].apply(lambda x: extract(x))
print(df[ (df['дв'].notna()) & (df['дв'] != df.numberOfDoors) ]['дв'].count())
df.drop(['дв'], inplace = True, axis = 1)
df[field_name] = df[field_name].apply(lambda x: str(x).split()[0])
describe(df, field_name)
describe(train, field_name)
a = set(train[train[field_name].notna()][field_name].unique().tolist())
b = set(test[field_name].unique().tolist())
print("Общие значения:", a.intersection(b))
print("Значения, которых нет в тесте:", a - b)
print("Значения, которых нет в трейне:", b - a)
def unify(x):
    if x == "лифтбек":
        return "хэтчбек"
    elif x == "компактвэн":
        return "минивэн"
    elif x == "родстер":
        return "кабриолет"
    else:
        return x
df[field_name] = df[field_name].apply(unify)
describe(df, field_name)
a = set(train[train[field_name].notna()][field_name].unique().tolist())
b = set(test[field_name].unique().tolist())
print("Общие значения:", a.intersection(b))
print("Значения, которых нет в тесте:", a - b)
print("Значения, которых нет в трейне:", b - a)
common_cols.add(field_name)
common_cols
field_name = 'brand'
describe(df, field_name)
describe(train, field_name)
common_cols.add(field_name)
common_cols
field_name = 'color'
describe(df, field_name)
describe(train, field_name)
train[field_name] = train[field_name].apply(lambda x: None if pd.isna(x) else x.lower())
describe(train, field_name)
common_cols.add(field_name)
common_cols
field_name = 'fuelType'
describe(df, field_name)
describe(train, field_name)
train[field_name] = train[field_name].apply(lambda x: None if pd.isna(x) else x.lower())
describe(train, field_name)
common_cols.add(field_name)
common_cols
field_name = 'modelDate'
describe(df, field_name)
describe(train, 'autocatalogWorldPremier')
def extract(x):
    found_year = re.search('(\d\d\d\d)', x)
    if found_year:
        return int(found_year.group(1))
    else:
        return None   
train[field_name] = train['autocatalogWorldPremier'].apply(lambda x: None if pd.isna(x) else extract(x))
describe(train, field_name)
field_name = 'name'
describe(df, field_name)
df['4WD'] = df[field_name].apply(lambda x: 1 if "4WD" in x else 0)
df[field_name] = df[field_name].apply(lambda x: x[:-4] if x.endswith(' 4WD') else x)
describe(df, field_name)
df.enginePower.unique()
A = 'л.с. из enginePower'
B = 'л.с. из name'
df[A] = df.enginePower.apply(lambda x: int(x.split()[0]))
def extract(x):
    found_engine_power = re.search('\((\d+)\s*л\.\с\.\)', x)
    if found_engine_power:
        return int(found_engine_power.group(1))
    else:
        return None
df[B] = df[field_name].apply(lambda x: extract(x))
print("Расхождений в лошадиных силах:", df[ (df[B].notna()) & (df[B] != df[A]) ][B].count())
df.drop([A, B], inplace = True, axis = 1)
def drop_suffix(x):
    x = re.sub('\s*\((\d+)\s*л\.\с\.\)$', '', x)
    return x
df[field_name] = df[field_name].apply(lambda x: drop_suffix(x))
describe(df, field_name)
df[(df[field_name].str.contains('кВт'))].enginePower
def drop_suffix(x):
    x = re.sub('\s*\((\d+)\s*кВт\)$', '', x)
    return x
df[field_name] = df[field_name].apply(lambda x: drop_suffix(x))
describe(df, field_name)
print("Трансмиссии: ", df.vehicleTransmission.unique())
df[df[field_name].str.contains("[AM]T", regex=True)][[field_name, 'vehicleTransmission']]
A = 'трансмиссия из поля name'
df[A] = df[field_name].apply(lambda x: 'автоматическая' if "AT" in x else 'механическая' if "MT" in x else None)
print("Расхождений в трансмиссии:", df[ (df[A].notna()) & (df[A] != df.vehicleTransmission) ][A].count())
print(df[ (df[A].notna()) & (df[A] != df.vehicleTransmission) ][['name', 'vehicleTransmission']])
A = 'трансмиссия из поля name'
def extract(x):
    if "AMT" in x:
        return 'роботизированная'  
    elif "AT" in x:
        return 'автоматическая'
    elif "MT" in x:
        return 'механическая'
    else:
        return None    
df[A] = df[field_name].apply(lambda x: extract(x))
print("Расхождений в трансмиссии:", df[ (df[A].notna()) & (df[A] != df.vehicleTransmission) ][A].count())
df.drop([A], inplace = True, axis = 1)
def drop_suffix(x):
    x = re.sub('\s[AM]+T$', '', x)
    return x
df[field_name] = df[field_name].apply(lambda x: drop_suffix(x))
describe(df, field_name)
def extract(x):
    found_suffix = re.search('\d+\.\d(\w+)$', x)
    if found_suffix:
        return found_suffix.group(1)
    else:
        return None
    
list(set(filter(lambda x: x is not None, map(extract, df[field_name].unique().tolist()))))
A = 'code'
df[A] = df[field_name].apply(lambda x: extract(x))
print(df[ (df[A].notna()) & (df[A] == 'd') ][[A, 'fuelType']])
print(df[ (df[A].notna()) & (df[A] == 'hyb') ][[A, 'fuelType']])
B = 'fuelType по коду'
df[B] = df[A].apply(lambda x: 'дизель' if x == 'd' else 'гибрид' if x == 'hyb' else None)
print("Расхождений в типе топлива:", df[ (df[B].notna()) & (df[B] != df.fuelType) ][B].count())
df.drop([A, B], inplace = True, axis = 1)
df[[field_name, 'engineDisplacement']]
A = 'объем двигателя из name'
B = 'объем двигателя из engineDisplacement'

def extract_engine_displacement(x, regex):
    found_engine_displacement = re.search(regex, x)
    if found_engine_displacement:
        return float(found_engine_displacement.group(1))
    else:
        return None
df[A] = df[field_name].apply(lambda x: extract_engine_displacement(x, '\s*(\d+\.\d)(\D.*)$'))
df[B] = df.engineDisplacement.apply(lambda x: extract_engine_displacement(x, '^(\d+\.\d)'))
print("Расхождений в объеме двигателя:", df[ (df[A].notna()) & (df[A] != df[B]) ][A].count())
df[ (df[A].notna()) & (df[A] != df[B]) ][[A, B]]
df.drop([A, B], inplace = True, axis = 1)
def drop_suffix(x):
    x = re.sub('\s*\d+\.\d\w*$', '', x)
    return x
df[field_name] = df[field_name].apply(lambda x: drop_suffix(x))
describe(df, field_name)
df[field_name].unique()
df['xDrive'] = df[field_name].apply(lambda x: 1 if "xDrive" in x else 0)
df['sDrive'] = df[field_name].apply(lambda x: 1 if "sDrive" in x else 0)
def drop_suffix(x):
    x = re.sub('[xs]Drive', '', x)
    return x
df[field_name] = df[field_name].apply(lambda x: drop_suffix(x))
describe(df, field_name)
df[field_name].unique()
df['model'] = df[field_name]

describe(train, field_name)
train[(train.brand == "BMW")]['autocatalogTitle'].unique()
df_temp = df
field_name_temp = field_name
try:
    df = train
    field_name = 'autocatalogTitle'
#     df[field_name] = train_orig[field_name]
    def drop_suffix(x):
        x = re.sub('\s*\((\d+)\s*л\.\с\.?\s*\)', '', x)
        x = re.sub('\s[AM]+T', '', x)
        return x
    df[field_name] = df[field_name].apply(lambda x: None if pd.isna(x) else drop_suffix(x))
    df['4WD'] = df[field_name].apply(lambda x: 1 if pd.notna(x) and 'AWD' in x else 0)
    df['FWD'] = df[field_name].apply(lambda x: 1 if pd.notna(x) and 'FWD' in x else 0)
    df['RWD'] = df[field_name].apply(lambda x: 1 if pd.notna(x) and 'RWD' in x else 0)
    df['4WD'] = df[field_name].apply(lambda x: 1 if pd.notna(x) and re.match(r'quattro|4(?:WD|MATIC|x?Motion)', x, flags=re.IGNORECASE) else 0)
    df['xDrive'] = df[field_name].apply(lambda x: 1 if pd.notna(x) and "xDrive" in x else 0)
    df['sDrive'] = df[field_name].apply(lambda x: 1 if pd.notna(x) and "sDrive" in x else 0)
    df['eDrive'] = df[field_name].apply(lambda x: 1 if pd.notna(x) and "eDrive" in x else 0)
    df['tronic'] = df[field_name].apply(lambda x: 1 if pd.notna(x) and ("tronic" in x or "Tronic" in x) else 0)
    def drop_suffix(x):
        x = re.sub('\s*[ARF]WD', '', x)
        x = re.sub('\s*4(x?motion|matic|wd)', '', x, flags=re.IGNORECASE)
        x = re.sub('\s*quattro', '', x)
        x = re.sub('\s*[xes]Drive', '', x)
        x = re.sub('\s*(?:Step|Tip|Multi)tronic', '', x)
        x = re.sub('\s*\d+G-Tronic', '', x, flags=re.IGNORECASE)
        return x
    df[field_name] = df[field_name].apply(lambda x: None if pd.isna(x) else drop_suffix(x))
    def drop_suffix(x):
        x = re.sub('\s*\d+\.\d\w*$', '', x)
        return x
    df[field_name] = df[field_name].apply(lambda x: None if pd.isna(x) else drop_suffix(x))
    print("Получившиеся модели BMW:", df[(df.brand == "BMW")][field_name].unique().tolist())
    print("------------------------")
    print("Получившиеся модели остальных брендов:", df[(df.brand != "BMW")][field_name].unique().tolist())
finally:
    df = df_temp
    field_name = field_name_temp
train['model'] = train['autocatalogTitle']
for field_name in ['model', 'xDrive', 'sDrive', '4WD']:
    common_cols.add(field_name)
common_cols
field_name = 'numberOfDoors'
describe(df, field_name)
describe(train, field_name)
common_cols.add(field_name)
common_cols
field_name = 'productionDate'
describe(df, field_name)
describe(train, field_name)
common_cols.add(field_name)
common_cols
field_name = 'vehicleConfiguration'
describe(df, field_name)
df[field_name].unique()
A = 'объем двигателя из vehicleConfiguration'
B = 'объем двигателя из engineDisplacement'
C = 'трансмиссия из vehicleConfiguration'
D = 'количество дверей из vehicleConfiguration'
E = 'тип кузова из vehicleConfiguration'
def extract_engine_displacement(x, regex):
    found_engine_displacement = re.search(regex, x)
    if found_engine_displacement:
        return float(found_engine_displacement.group(1))
    else:
        return None
df[A] = df[field_name].apply(lambda x: extract_engine_displacement(x, '\s*(\d+\.\d)(\D.*)$'))
df[B] = df.engineDisplacement.apply(lambda x: extract_engine_displacement(x, '^(\d+\.\d)'))
print("Расхождений в объеме двигателя:", df[ (df[A].notna()) & (df[A] != df[B]) ][A].count())
print("Варианты трансмиссии из vehicleConfiguration:", set(list(map(lambda x: x.split()[1], df[field_name].unique()))))
def extract_vehicle_transmission(x):
    ss = x.split()
    if len(ss) < 2:
        return None
    else:
        s = ss[1]
        if s == 'ROBOT':
            return 'роботизированная'
        elif s == 'MECHANICAL':
            return 'механическая'
        elif s == 'AUTOMATIC':
            return 'автоматическая'
        else:
            return 'Unknown'
df[C] = df[field_name].apply(lambda x: extract_vehicle_transmission(x))
print("Расхождений в трансмиссии:", df[ (df[C].notna()) & (df[C] != df.vehicleTransmission) ][C].count())
def extract_number_of_doors(x):
    ss = x.split()
    if len(ss) < 1:
        return None
    else:
        s = ss[0]
        found_number_of_doors = re.search('_(\d+)_DOORS', s)
        if found_number_of_doors:
            return int(found_number_of_doors.group(1))
        else:
            return None
df[D] = df[field_name].apply(lambda x: extract_number_of_doors(x))
print("Расхождений в количестве дверей:", df[ (df[D].notna()) & (df[D] != df.numberOfDoors) ][D].count())
df[D].unique()
def extract_body_type(x):
    ss = x.split()
    if len(ss) < 1:
        return None
    else:
        s = ss[0]
        ss = s.split('_')
        if len(ss) < 1:
            return None
        else:
            return ss[0]
print("Варианты типа кузова из vehicleConfiguration:", 
      set(list(map(lambda x: extract_body_type(x), df[field_name].unique())))
)
def extract_body_type_rus(x):
    s = extract_body_type(x)
    if s is None:
        return None
    elif s == 'ROADSTER':
        return 'родстер'
    elif s == 'HATCHBACK':
        return 'хэтчбек'
    elif s == 'LIFTBACK':
        return 'лифтбек'
    elif s == 'COMPACTVAN':
        return 'компактвэн'
    elif s == 'CABRIO':
        return 'кабриолет'
    elif s == 'SEDAN':
        return 'седан'
    elif s == 'COUPE':
        return 'купе'
    elif s == 'WAGON':
        return 'универсал'
    elif s == 'ALLROAD':
        return 'внедорожник'
    else:
        return 'Unknown'
df[E] = df[field_name].apply(lambda x: extract_body_type_rus(x))
print("Расхождений в типе кузова:", df[ (df[E].notna()) & (df[E] != df.bodyType) ][E].count())
df.drop([A, B, C, D, E], inplace = True, axis = 1)
df.drop([field_name], inplace = True, axis = 1)
field_name = 'vehicleTransmission'
describe(df, field_name)
describe(train, field_name)
train[field_name] = train[field_name].apply(lambda x: 'роботизированная' if pd.notna(x) and x == 'робот' else x)
describe(train, field_name)
common_cols.add(field_name)
common_cols
field_name = 'engineDisplacement'
describe(df, field_name)
df[field_name] = df[field_name].apply(lambda x: x.split()[0])
describe(df, field_name)
df[df[field_name] == 'undefined']
describe(train, field_name)
train[field_name] = train[field_name].apply(lambda x: "6.0" if pd.notna(x) and x == "6.0+" else x)
describe(train, field_name)
common_cols.add(field_name)
common_cols
field_name = 'enginePower'
describe(df, field_name)
df[field_name] = df[field_name].apply(lambda x: x.split()[0])
describe(df, field_name)
describe(train, field_name)
def drop_suffix(x):
    x = re.sub(r'\s*л.с\.?\s*$', '', x)
    return x
train[field_name] = train[field_name].apply(lambda x: None if pd.isna(x) else drop_suffix(x))
train[field_name].unique()
common_cols.add(field_name)
common_cols
field_name = 'description'
describe(df, field_name)
df.drop([field_name], inplace = True, axis = 1)
field_name = 'mileage'
describe(df, field_name)
describe(train, 'mileage')
common_cols.add(field_name)
common_cols
field_name = 'Комплектация'
describe(df, field_name)
import json
features = set()
for s in df[field_name].unique().tolist():
    if len(s) > 4:
        decoded = json.loads(s[2:-2])
        for segment in decoded:
            name = segment['name']
            for feature in segment['values']:
                features.add(feature + '::' + name)
print("Количество разных атрибутов комлектации:", len(features))
print("Список атрибутов комлектации:")
print("------------------------------")
for feature in sorted(list(features)):
    print(feature)
def extract_feature(feature, s):
    (feature_name, segment_name) = feature.split('::')
    if len(s) > 4:
        decoded = json.loads(s[2:-2])
        for segment in decoded:
            if segment['name'] == segment_name:
                for feature in segment['values']:
                    if feature == feature_name:
                        return 1
    return 0
for feature in sorted(list(features)):
    df[feature] = df[field_name].apply(lambda x: 0 if pd.isna(x) else extract_feature(feature, x))
    
df.info(verbose=True)
for feature in ['Яндекс.Авто::Мультимедиа', 'Фаркоп::Прочее', 'Сиденья с массажем::Салон']:
    print(df[feature].value_counts())
    print("")
describe(train, 'Электростеклоподъемники')
feature = 'Электростеклоподъёмники'
source = 'Электростеклоподъемники'
group = 'Комфорт'
for kind in ["задние", "передние"]:
    field_name = f'{feature} {kind}::{group}'
    train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and kind in x else 0)
    describe(train, field_name)
    common_cols.add(field_name)
    
    print("")
describe(train, 'Усилитель руля')
feature = 'Усилитель руля'
group = 'Комфорт'
source = feature
field_name = f'{feature}::{group}'
train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) else 0)
describe(train, field_name)
common_cols.add(field_name)
describe(train, 'Аудиосистема')
describe(train, 'Фары')
source = 'Фары'
group = 'Обзор'

feature = 'Ксеноновые/Биксеноновые фары'
field_name = f'{feature}::{group}'
mark = 'ксеноновые'
train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and mark in x else 0)
common_cols.add(field_name)
describe(train, field_name)
print("")

feature = 'Светодиодные фары'
field_name = f'{feature}::{group}'
mark = 'светодиодные'
train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and mark in x else 0)
common_cols.add(field_name)
describe(train, field_name)

describe(train, 'Климат-контроль')
source = 'Климат-контроль'
group = 'Комфорт'

feature = 'Климат-контроль многозонный'
field_name = f'{feature}::{group}'
mark = 'многозонный'
train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and mark in x else 0)
common_cols.add(field_name)
describe(train, field_name)
print("")

feature = 'Климат-контроль 1-зонный'
field_name = f'{feature}::{group}'
mark = 'однозонный'
train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and mark in x else 0)
common_cols.add(field_name)
describe(train, field_name)

describe(train, 'Салон')
source = 'Салон'
group = 'Салон'

feature = 'Ткань (Материал салона)'
field_name = f'{feature}::{group}'
mark = 'ткань'
train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and mark in x else 0)
common_cols.add(field_name)
describe(train, field_name)
print("")

feature = 'Кожа (Материал салона)'
field_name = f'{feature}::{group}'
mark = 'кожа'
train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and mark in x else 0)
common_cols.add(field_name)
describe(train, field_name)
print("")

feature = 'Велюр (Материал салона)'
field_name = f'{feature}::{group}'
mark = 'велюр'
train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and mark in x else 0)
common_cols.add(field_name)
describe(train, field_name)
print("")

feature = 'Комбинированный (Материал салона)'
field_name = f'{feature}::{group}'
mark = 'комбинированный'
train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and mark in x else 0)
common_cols.add(field_name)
describe(train, field_name)
describe(train, 'Диски')
source = 'Диски'
group = 'Элементы экстерьера'

for i in range(14,23):
    feature = f'Диски {i}'
    field_name = f'{feature}::{group}'
    mark = str(i)
    train[field_name] = train[source].apply(lambda x: 1 if pd.notna(x) and mark in x else 0)
    common_cols.add(field_name)
    describe(train, field_name)
field_name = 'Привод'
describe(df, field_name)
describe(train, field_name)
common_cols.add(field_name)
field_name = 'Руль'
describe(df, field_name)
describe(train, field_name)
common_cols.add(field_name)
field_name = 'Состояние'
describe(df, field_name)
describe(train, field_name)
train = train[ train[field_name].map(lambda x: 1 if pd.notna(x) and x == "Битый" else 0) == 0 ]
describe(train, field_name)
train.info(verbose=True)
field_name = 'Владельцы'
describe(df, field_name)
describe(train, field_name)
df[field_name] = df[field_name].apply(lambda x: x[0])
train[field_name] = train[field_name].apply(lambda x: None if pd.isna(x) else x[0] if int(x[0]) < 4 else "3")
describe(df, field_name)
describe(train, field_name)
common_cols.add(field_name)
field_name = 'ПТС'
describe(df, field_name)
field_name = 'Таможня'
describe(df, field_name)
field_name = 'Владение'
describe(df, field_name)
df[field_name].unique()
def transform(x):
    if x is None:
        return None
    else:
        found_month = re.search('^(\d+) месяц(?:а|ев)?', x)
        if found_month:
            return found_month.group(1)   
        else:
            found_year_month = re.search('^(\d+) (?:года?|лет)(?: и (\d+) месяц(?:а|ев)?)?', x)
            if found_year_month is None:
                print(x)
                return None
            else:
                if found_year_month.group(2) is None:                   
                    return int(found_year_month.group(1)) * 12
                else:
                    return int(found_year_month.group(1)) * 12 + int(found_year_month.group(2))
df[field_name] = df[field_name].apply(lambda x: None if pd.isna(x) else transform(x))
print("Общих полей:", len(common_cols))
print("------------")
for field_name in sorted(list(common_cols)):
    print(field_name)
field_name = 'is_train'
test[field_name] = 0
train[field_name] = 1
common_cols.add(field_name)
field_name = 'price'
test[field_name] = None
train[field_name] = train['itemPrice']
common_cols.add(field_name)
common_cols = sorted(list(common_cols))
common_df_orig = test[common_cols].append(train[common_cols], sort=False).reset_index(drop=True)
print(common_df_orig.info(verbose=True))
df = common_df_orig
describe(df, 'fuelType')
field_name = 'Налог'
def calc_tax(x):
    x = int(x)
    return x*12 if x<=100 else x*25 if x>100 and x<=125 else x*35 if x>125 and x<=150 else x*45 if x>150 and x<=175 else x*55 if x>175 and x<=200 else x*65 if x>200 and x<=225 else x*75 if x>225 and x<=250 else x*150
df[field_name] = df.apply(lambda row: 
                          0 if row['fuelType'] == 'электро' else 
                          None if pd.isna(row['enginePower']) else calc_tax(row['enginePower']), 
                            axis=1
                         )
# df[]
describe(df, field_name)
field_name = 'Время эксплуатации'
source = 'productionDate'
df[field_name] = 2021 - df[source]
describe(df, field_name)
field_name = 'Средний пробег'
df[field_name] = df['mileage'].astype(float) / df['Время эксплуатации'].astype(float)
field_name = 'engineDisplacement'
df[field_name] = df[field_name].apply(lambda x: 0 if pd.isna(x) or x == 'undefined' else int(float(x) * 10))
describe(df, field_name)
field_name = 'price'
df[field_name] = df[field_name].fillna(0).astype(np.int)

field_name = 'engineDisplacement'

df = df.dropna()

for field_name in [
    'enginePower', 
    'mileage', 
    'numberOfDoors', 
    'productionDate', 
    'Владельцы',
    'Налог', 
    'Время эксплуатации', 
    'Средний пробег'
]:
    df[field_name] = df[field_name].astype(np.int)
common_df_orig = df
sample_submission_url = "https://drive.google.com/u/0/uc?id=1XktGbf7aLmAd_eyTBL0YGYGXV8WLN5e1&export=download"
sample_submission_orig = pd.read_csv(sample_submission_url)
def make_submission(model, version, tag):
    predict_submission = model.predict(test)
    sample_submission = sample_submission_orig.copy()
    sample_submission['price'] = predict_submission
    sample_submission.price = sample_submission.price.astype(np.int32)
    sample_submission.to_csv(f'submission_v{version}_{tag}.csv', index=False)
    sample_submission.head(10)
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred1 = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f'RMSE = {rmse:.2f}, MAE = {mae:.2f}, R-sq = {r2:.2f}, MAPE = {mape:.2f} ')
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df = common_df_orig.copy()
for field_name in ['bodyType', 'brand', 'color', 'fuelType', 'vehicleTransmission', 'Привод', 'Руль', 'model']:
    v = df[field_name].values.tolist()
    encoder.fit(v)
    df[field_name] = encoder.transform(df[field_name])
    describe(df, field_name)
train = df.query('is_train==1').drop(['is_train'], axis = 1)
test = df.query('is_train==0').drop(['is_train', 'price'], axis = 1)
VAL_SIZE = 0.3
RANDOM_SEED = 77
from sklearn.model_selection import train_test_split
target_field_name = 'price'
X = train.drop([target_field_name],axis=1)
y = train[target_field_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size = VAL_SIZE, random_state=RANDOM_SEED)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

RANDOM_SEED = 42
model = RandomForestRegressor(n_estimators=30, max_features=10,  max_depth = 20, random_state=RANDOM_SEED) #max_depth = 15, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 100, verbose = 100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print_regression_metrics(y_test, y_pred)
make_submission(model, VERSION, 'random_forest')
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df = common_df_orig.copy()
for field_name in ['bodyType', 'brand', 'color', 'fuelType', 'vehicleTransmission', 'Привод', 'Руль', 'model']:
    v = df[field_name].values.tolist()
    encoder.fit(v)
    df[field_name] = encoder.transform(df[field_name])
    describe(df, field_name)
# https://stackoverflow.com/questions/14173421/use-string-translate-in-python-to-transliterate-cyrillic
symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ :()",
           u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA____")
tr = {ord(a):ord(b) for a, b in zip(*symbols)}
rename_map = {}
for col in df.columns.tolist():
    field_name = col.translate(tr)
    rename_map[col] = col.translate(tr)
df.rename(columns=rename_map, inplace=True)

train = df.query('is_train==1').drop(['is_train'], axis = 1)
test = df.query('is_train==0').drop(['is_train', 'price'], axis = 1)
VAL_SIZE = 0.3
RANDOM_SEED = 77
from sklearn.model_selection import train_test_split
target_field_name = 'price'
X = train.drop([target_field_name],axis=1)
y = train[target_field_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size = VAL_SIZE, random_state=RANDOM_SEED)

import lightgbm as lgb

model = lgb.LGBMRegressor(random_state=RANDOM_SEED)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print_regression_metrics(y_test, y_pred)
make_submission(model, VERSION, 'ligthgbm')
df = common_df_orig.copy()
train = df.query('is_train==1').drop(['is_train'], axis = 1)
test = df.query('is_train==0').drop(['is_train', 'price'], axis = 1)
from sklearn.model_selection import train_test_split

VAL_SIZE   = 0.30
RANDOM_SEED = 42

target_field_name = 'price'
X = train.drop([target_field_name],axis=1)
y = train[target_field_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size = VAL_SIZE, random_state=RANDOM_SEED)
from sklearn.linear_model import LinearRegression
from sklearn import metrics 
from catboost import CatBoostRegressor

# CATBOOST
ITERATIONS = 10000
LR         = 0.1
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
model.save_model('catboost_single_model_baseline.model')
y_pred = model.predict(X_test)
print_regression_metrics(y_test, y_pred)
features_importances = pd.DataFrame(data = model.feature_importances_, index = X.columns, columns = ['FeatImportant'])
features_importances.sort_values(by = 'FeatImportant', ascending = False).head(20)
make_submission(model, VERSION, 'catboost')
N_FOLDS    = 10
from sklearn.model_selection import KFold
from tqdm import tqdm

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
sample_submission = sample_submission_orig.copy()

submissions = pd.DataFrame(0,columns=["sub_1"], index=sample_submission.index) # куда пишем предикты по каждой модели
score_ls = []
splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, y))
for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total=N_FOLDS,):
    # use the indexes to extract the folds in the train and validation data
    X_train, y_train, X_test, y_test = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]
    # model for this fold
    model = cat_model(y_train, X_train, X_test, y_test,)
    # score model on test
    test_predict = model.predict(X_test)
    test_score = mape(y_test, test_predict)
    score_ls.append(test_score)
    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")
    # submissions
    submissions[f'sub_{idx+1}'] = model.predict(test)
    model.save_model(f'catboost_fold_{idx+1}.model')
print(f'Mean Score: {np.mean(score_ls):0.3f}')
print(f'Std Score: {np.std(score_ls):0.4f}')
print(f'Max Score: {np.max(score_ls):0.3f}')
print(f'Min Score: {np.min(score_ls):0.3f}')
y_pred = model.predict(X_test)
print_regression_metrics(y_test, y_pred)
make_submission(model, VERSION, 'catboost_blended')
