import pandas as pd

import pandas_profiling

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import re

from sklearn.feature_selection import f_classif, mutual_info_classif

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.tree import ExtraTreeRegressor

from tqdm import tqdm

from itertools import combinations

from scipy.stats import ttest_ind

from catboost import CatBoostClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None



import os



# этот блок закомментирован так как используется только на kaggle

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

PATH_to_file = '/kaggle/input/sf-dst-car-price/'

PATH_to_file_data = '/kaggle/input/parsing-all-moscow-auto-ru-09-09-2020/'



# # # этот блок закомментирован так как используется только локальной машине

# from importlib import reload

# print(os.listdir('./data'))

# PATH_to_file = './data/'
import utils_module09092020 as utils
RANDOM_SEED = 42

!pip freeze > requirements.txt

CURRENT_DATE = pd.to_datetime('19/09/2020')
df_train = pd.read_csv(PATH_to_file_data +'all_auto_ru_09_09_2020.csv')

df_test = pd.read_csv(PATH_to_file+'test.csv')

df_submit = pd.read_csv(PATH_to_file+'sample_submission.csv')

pd.set_option('display.max_columns', None)

print('Размерность тренировочного датасета: ', df_train.shape)

display(df_train.head(2))

print('Размерность тестового датасета: ', df_test.shape)

display(df_test.head(2))

print('Размерность датасета c примером сабмишена: ', df_submit.shape)

display(df_submit.head(2))
# сравним датасеты 

utils.check_df_before_merg(df_train, df_test)
# починим датасет перед использованием и приведем его к виду теста



# на первый взгляд может, что перевод во float достаточно бесмыссленный шаг ведь кол-во дверей автомобиля это целое число. Но напомню что мы сейчас на этапе приведения датасета после парсинга к виду тестового. На этапе анализа датасета мы сможем более детально посмотреть что происходит в тестовом датасете (вероятно там есть ошибки которые надо исправить и мы можем потерять эту информацию не знаю сколько ошибок и какие они) 



# начнем с int64!=float64 и приведем все к float 

list_cols_to_repair = ['productionDate', 'mileage']

for col in list_cols_to_repair:

        df_train[col] = df_train[col].astype('float64')



# теперь починим int64!=float64 и приведем все к object

list_cols_to_repair = ['enginePower', 'Состояние', 'Владельцы']

for col in list_cols_to_repair:

        df_train[col] = df_train[col].astype('object')



# теперь починим bool!=object

# предварительно райдем в датасет и убедимся, что True соответсвует Растаможен зайдя на сайте (этот код тут не приводится, чтобы не нагружать ноутбук)

df_train['Таможня'] = df_train['Таможня'].map({True: 'Растаможен', False:'Не растаможен'})
utils.check_df_before_merg(df_train, df_test)
# bodyType - надо привести значения к нижнему регистру (оставим закадром, проверку уникальных значений трейна и теста, чтобы прийти к такому выводу)

df_train['bodyType'] = df_train['bodyType'].apply(lambda x: str(x).lower())

utils.nunique_not_found(df_train, df_test, 'bodyType')





# color - цвета спарсились в шестнадцатеричном коде цветов. Но их всего 16 поэтому воспользуемся сайтом (https://hysy.org/color) и сделаем словарь цветов. (оставим закадром, проверку уникальных значений трейна и теста, чтобы сделать соответсвия цветов) 

dict_color = {'040001':'чёрный', 'EE1D19':'красный', '0000CC':'синий', 

              'CACECB':'серебристый', '007F00':'зелёный', 'FAFBFB':'белый', 

              '97948F':'серый', '22A0F8':'голубой', '660099':'пурпурный', 

              '200204':'коричневый', 'C49648':'бежевый', 'DEA522':'золотистый', 

              '4A2197':'фиолетовый', 'FFD600':'жёлтый', 'FF8649':'оранжевый', 

              'FFC0CB':'розовый'}

df_train['color'] = df_train['color'].map(dict_color)

utils.nunique_not_found(df_train, df_test, 'color')



# vehicleTransmission - тип трасмиссии по словарю из 4 значений

df_train['vehicleTransmission'] = df_train['vehicleTransmission'].map({'MECHANICAL':'механическая', 'AUTOMATIC':'автоматическая', 'ROBOT':'роботизированная', 'VARIATOR':'вариатор'})





# Руль - ну тут все просто - Правый или Левый в нижнем регистре с большой буквы

df_train['Руль'] = df_train['Руль'].map({'RIGHT':'Правый', 'LEFT':'Левый'})

utils.nunique_not_found(df_train, df_test, 'Руль')



# ПТС - тут все просто - Оригинал или Дубликат в нижнем регистре с большой буквы

df_train['ПТС'] = df_train['ПТС'].map({'ORIGINAL':'Оригинал', 'DUPLICATE':'Дубликат'})

utils.nunique_not_found(df_train, df_test, 'ПТС')



# Владельцы - Оригинал или Дубликат в нижнем регистре с большой буквы

df_train['Владельцы'] = df_train['Владельцы'].map({3.0:'3 или более', 2.0:'2\xa0владельца', 1.0:'1\xa0владелец'})

utils.nunique_not_found(df_train, df_test, 'Владельцы')



# Владение - в таблице сверху в первых строках Nan поэтому нужно отдельно проверить

# оказалось что в тесте очень сложная строковая запись тогда как в трейне в значениях более структурируемые словари (УчТЕМ этот момент и разберем этот столбец на этапе анализа, вероятно будет сразу полезно создать дополнительные временные фичи) 

utils.nunique_not_found(df_train, df_test, 'Владение');
# ВАЖНО! для корректной обработки признаков объединяем трейн и тест в один датасет

df_train['Train'] = 1 # помечаем где у нас трейн

df_test['Train'] = 0 # помечаем где у нас тест



df = df_train.append(df_test, sort=False).reset_index(drop=True) # объединяем

#!Обратите внимание объединение датасетов является потенциальной опасностью для даталиков
# проверка  после слияния

df.head(2)
# # этот блок закомментирован так как pandas_profiling некорректно отображается на kaggle

# # анализ тренировочной части

# pandas_profiling.ProfileReport(df[df['Train']==1])
# # этот блок закомментирован так как pandas_profiling некорректно отображается на kaggle

# # анализ тестовой части

# pandas_profiling.ProfileReport(df[df['Train']==0])
# выведем сводную информацию по датасету df без теста kaggle

utils.describe_without_plots_all_collumns(df[df['Train']==1], short=True)
# выведем сводную информацию по тесту

utils.describe_without_plots_all_collumns(df[df['Train']==0], short=True)
# внесем данные по типам переменных из резюме в списки 

# временной ряд (1)

time_cols = ['start_date']

# бинарная переменная (0) (hidden - не включаем так как решили удалить) итого  (0+1=1)

bin_cols = []

# категориальные переменные (19), ('Таможня' - не включаем  так как решили удалить) (19+1=20)

cat_cols = ['bodyType', 'brand', 'color', 'fuelType', 'name',

       'numberOfDoors', 'vehicleConfiguration',

       'vehicleTransmission', 'engineDisplacement', 'enginePower',

       'description', 'Комплектация', 'Привод', 'Руль', 'Состояние',

       'Владельцы', 'ПТС', 'Владение', 'model']

# числовые переменные (3) 

num_cols = ['mileage', 'modelDate', 'productionDate']

# сервисные переменные (2)

servis_cols = ['Train', 'id']

# целевая переменная (1)

target_col = ['price']

# итого 1+1+20+3+2+1=28



all_cols =cat_cols+num_cols+time_cols+servis_cols+bin_cols+ target_col

print(f'Кол-во столбцов, для дальнейшей работы после предварительного анализа:= {len(all_cols)}')
# реализуем выводы из резюме перед детальным анализом по переменным

# берем только левый руль и тольк 26 столбцов

df = df.loc[df['Руль'] == 'Левый', all_cols]

# исключаем Руль из списка признаков для анализа итого остается 25 признаков из которых 2 сервисных (нужно проанализировать 23=25-2)

cat_cols.remove('Руль')

# удаляем дубликаты

df = df.drop_duplicates()

old_len_train = len(df[df['Train']==1])

print('Кол-во строк в трейне:= ', old_len_train)
# так как признаков много создаем список проанализированных признаков, чтобы можно было посмотреть, что осталось сделать  

EDA_done_cols=[]
temp_df_Train = df[df['Train']==1]

temp_df_Test = df[df['Train']==0]

print(f'Всего в тесте типов кузова:= {temp_df_Test.bodyType.nunique()}')

print(f'Всего в трейне типов кузова:= {temp_df_Train.bodyType.nunique()}')
# посмотрим на них 

print(f'Список 11 типов кузова БМВ из теста:= {temp_df_Test.bodyType.unique()}')

print(f'Список первых 20 из 144 типов кузова из трейна:= {temp_df_Train.bodyType.unique()[:20]}')
# оставляем только типы кузова как в тесте

list_bodyType_test = list(temp_df_Test.bodyType.unique())

df = df[df['bodyType'].isin(list_bodyType_test)]
old_len_train, EDA_done_cols = utils.result_EDA_feature('bodyType', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# вспомним что по результатам предварительного анализа эта переменная была отнесена к категориальнымu, посмотрим почему

temp_df_Train = df[df['Train']==1]

temp_df_Test = df[df['Train']==0]

display(temp_df_Train.enginePower[:5])

display(temp_df_Test.enginePower[:5])
# проверим все значения теста по мощности 

temp_df_Test.enginePower.unique()
# в тесте вытащим значение мощности перед строкой ' N12'

df.loc[df['Train']==0, 'enginePower'] = df[df['Train']==0]['enginePower'].apply(lambda x: int(x.split()[0]))
# переводим в int64

df['enginePower'] = df['enginePower'].astype('int64')



# перенесем признак в числовые

cat_cols.remove('enginePower')

num_cols.append('enginePower')
utils.describe_without_plots('enginePower', df[df['Train']==1].enginePower)
# очень странное минимальное значение можности, давайте посмотрим минимум мощности на тесте 

utils.describe_without_plots('enginePower', df[df['Train']==0].enginePower)
df[(df['Train']==1) & (df['enginePower']<90)]['brand'].value_counts().plot(kind = 'barh', title='Cтатистика по маркам автомобилей с мощностью менее 90 л.с.')
df[(df['Train']==1) & (df['enginePower']>625)]['brand'].value_counts().plot(kind = 'barh', title='Cтатистика по маркам автомобилей с мощностью более 625 л.с.')
# удалим значения мощностей меньше 90 и выше 625

df=df[(df['enginePower']<=625) & (df['enginePower']>=90)]
utils.four_plot_with_log2('enginePower', df[df['Train']==1])
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('enginePower', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
print(f'Всего в датасета марок:= {df.brand.nunique()}')
list_brand0 = utils.hbar_group_pivot_table(list_bodyType_test[0], 'price', df[df['Train']==1], 2013, 2020, 1.1)
list_brand1 = utils.hbar_group_pivot_table(list_bodyType_test[1], 'price', df[df['Train']==1], 2015, 2019, 1.1)
list_brand2 = utils.hbar_group_pivot_table(list_bodyType_test[2], 'price', df[df['Train']==1], 2000, 2010, 1.1)
list_brand3 = utils.hbar_group_pivot_table(list_bodyType_test[3], 'price', df[df['Train']==1], 2015, 2019, 1.1)
list_brand4 = utils.hbar_group_pivot_table(list_bodyType_test[4], 'price', df[df['Train']==1], 2015, 2019, 1.1)
list_brand5 = utils.hbar_group_pivot_table(list_bodyType_test[5], 'price', df[df['Train']==1], 1990, 2010, 1.1)
list_brand7 = utils.hbar_group_pivot_table(list_bodyType_test[7], 'price', df[df['Train']==1], 2000, 2005, 1.1)
list_brand8 = utils.hbar_group_pivot_table(list_bodyType_test[8], 'price', df[df['Train']==1], 2000, 2010, 1.1)
list_brand10 = utils.hbar_group_pivot_table(list_bodyType_test[10], 'price', df[df['Train']==1], 2015, 2019, 1.1)
# список всех релевантных моделей

list_final_brand=list(set(list_brand0+list_brand1+list_brand2+list_brand3+list_brand4+list_brand5+list_brand7+list_brand8+list_brand10))

print('Список релевантных брендов авто:\n',*list_final_brand, '\n, их кол-во:=', len(list_final_brand), 'из 36')

print()
# кроме этого в трейне нет некоторых авто БМВ с типом кузова

print('В трейне не оказалось авто БМВ с типом кузова:=', list_bodyType_test[6])

temp_df_Train = df[df['Train']==1]

temp=list(temp_df_Train[temp_df_Train['bodyType']==list_bodyType_test[6]].brand.unique())

print('Список брендов авто с типом кузова', list_bodyType_test[6], ':=', *temp)
# кроме этого в трейне нет некоторых авто БМВ с типом кузова

print('В трейне не оказалось авто БМВ с типом кузова:=', list_bodyType_test[9])

temp_df_Train = df[df['Train']==1]

temp=list(temp_df_Train[temp_df_Train['bodyType']==list_bodyType_test[9]].brand.unique())

print('Список брендов авто с типом кузова', list_bodyType_test[9], ':=', *temp)
temp_df_Train[temp_df_Train['bodyType']==list_bodyType_test[9]]
# оставляем 22 релевантных (21+BMW) бренд авто, остальные удаляем

df = df[df['brand'].isin(list_final_brand+['BMW'])]
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('brand', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# вспомним что по результатам предварительного анализа в этой переменной было 6 уникальных значений в трейне и 4 в тесте

temp_df_Train = df[df['Train']==1]

temp_df_Test = df[df['Train']==0]

print(f'Список зачений по fuelType в трейне:= {list(temp_df_Train.fuelType.unique())}')

print(f'Список зачений по fuelType в тесте:= {list(temp_df_Test.fuelType.unique())}')
# уберем газ так как автомобилей на газе нет в тесте

list_fuelType_test = temp_df_Test.fuelType.unique()

df = df[df['fuelType'].isin(list_fuelType_test)]
# Посмотрим сколько элекромобилей в тесте

print('Распределение по кол-ву зачений по fuelType в трейне:= \n',temp_df_Train.fuelType.value_counts())

print('Распределение по кол-ву зачений по fuelType в тесте:= \n',temp_df_Test.fuelType.value_counts())
# посмотрим на автомобили БМВ в трейне и в тесте

temp_df = temp_df_Train[(temp_df_Train['fuelType']=='электро')&(temp_df_Train['brand']=='BMW')]

print(f'Кол-во электрокаров BMW в трейне:= {len(temp_df)}')

display(temp_df)



display(temp_df_Test[(temp_df_Test['fuelType']=='электро')&(temp_df_Test['brand']=='BMW')])
# посмотрим какие бренды в остальных (68-7=61) электрокаров трейна и как распределены по ним цены

temp_df = temp_df_Train[(temp_df_Train['fuelType']=='электро')] 

list_brand = utils.hbar_group_pivot_table(list_bodyType_test[3], 'price', temp_df, 2010, 2019, 1.1)
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('fuelType', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# обработаем значения с помощью регулярных выражений найдя значения объема в литра типа 2.0 и переведем их см3 

def engineDisplacement_to_float(row):

    row = str(row)

    volume = re.findall('\d\.\d', row)

    if volume == []:

        return None

    return int(float(volume[0])*1000)

# поле engineDisplacement заполненно не полностью именно мощностью, в отличии от поля name

# поэтому вытаскиваем мощность из поля name

df['engineDisplacement2'] = df['name'].apply(engineDisplacement_to_float)
# посмотрим сколько пропусков

len(df[df['engineDisplacement2'].isna()])
# подозрительно знакомое число видимо это электрокары, проверим 

df[df['engineDisplacement2'].isna()].fuelType.unique()
# посмотрим что там внутри 

temp_df_Train = df[df['Train']==1]

temp_df = temp_df_Train[temp_df_Train['engineDisplacement2'].isna() & (temp_df_Train['brand']=='BMW')]

display(temp_df.head(2))

display(temp_df.describe())
# чтож заполним пропуски аналогичными объемами двигателей автомобилей со следующими параметрами:

# 1 bodyType == хэтчбек 5 дв.

# 2 modelDate от 2013 до 2017

# 3 enginePower = около 170

# 4 price = 2.017469e+06	+ std(3.601556e+05) = (1657314, 2377624) 

# 5 brand == BMW

# 6 fuelType != 'электро'



temp_df = temp_df_Train[(temp_df_Train['bodyType']==list_bodyType_test[3]) & (temp_df_Train['brand']=='BMW') & (temp_df_Train['modelDate']>=2013) & (temp_df_Train['modelDate']<=2017)& (temp_df_Train['price']>=1657314) & (temp_df_Train['modelDate']<=2377624)& (temp_df_Train['enginePower']>=170-10) & (temp_df_Train['enginePower']<=170+10)& (temp_df_Train['fuelType']!='электро')]

display(temp_df.head(2))

display(temp_df.describe())
# похоже что с такой низкой мощностью близкиеми оказались только гибриды, но 67 пропусков не так много - заполним мощность 700 см3, это на самом деле не объем двигателя гибрида действительно около 700 см3 мощностью 34 л.с, к которым добавляется 170 л.с электромотра в случае экземпляров из первых двух строк выше. И правильно было бы перевести их в усредненный объем двигателя соответсвующий 200 л.с., но с другой стороны все гибриды и электрокары будут с таким маленьким объемом и возможно он сможет найти закономерность и понять что гибриды и электрокары дороже своих аналогов 

df['engineDisplacement2'].fillna(700, inplace = True)
# добавим мощность двигателя engineDisplacement2 в числовые признаки, а engineDisplacement удалим

cat_cols.remove('engineDisplacement') 

num_cols.append('engineDisplacement2')
utils.describe_without_plots('engineDisplacement2', df[df['Train']==1].engineDisplacement2)
# надо посмотреть где получились нули

df[df['engineDisplacement2']==0].head(5)
# это два электрокара 2019 года выпуска с ценой выше в 2 раза средней по BMW

# удаляем

df = df[df['engineDisplacement2']!=0]
utils.describe_without_plots('engineDisplacement2', df[df['Train']==1].engineDisplacement2)
# а какой максимум по BMW в тесте

temp_df = df[df['Train']==0 & (df['brand']=='BMW')]

utils.describe_without_plots('engineDisplacement2', temp_df.engineDisplacement2)
# какие авто с объемом двигателя более 6600 см3

temp_df= df[df['engineDisplacement2']>6600]

display(temp_df.head(3))

print(f'Кол-во авто с объемом двигателя более 6600 см3:= {len(temp_df)}')

print(f'Бренды авто с объемом двигателя более 6600 см3:= {list(temp_df.brand.unique())}')
# удаляем эти 9

df = df[df['engineDisplacement2']<=6600]
utils.four_plot_with_log2('engineDisplacement2', df[df['Train']==1])
# добавим новый признак логарифм

df['engineDisplacement2_log'] = np.log(df['engineDisplacement2'] + 1)

num_cols.append('engineDisplacement2_log')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('engineDisplacement', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# переведем в int64 критерий numberOfDoors

df['numberOfDoors'] = df['numberOfDoors'].astype('int64')
# посмотрим на зависимсть среднего значения цены от кол-ва дверей

df.groupby('numberOfDoors').mean().sort_values(by = 'price').price
utils.describe_without_plots('numberOfDoors', df[df['Train']==1].numberOfDoors)
utils.four_plot_with_log2('numberOfDoors', df[df['Train']==1])
# фактически это категориальный признак с 4 значениями, но для корректной групповой обработки числовых признаков переведем его в числовые

cat_cols.remove('numberOfDoors')

num_cols.append('numberOfDoors')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('numberOfDoors', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# переведем в int64 критерий numberOfDoors

df['mileage'] = df['mileage'].astype('int64')
utils.describe_without_plots('mileage', df[df['Train']==1].mileage)
utils.four_plot_with_log2('mileage', df[df['Train']==1])
# посмотрим гистограммы на не новых автомобилях

utils.four_plot_with_log2('mileage', df[(df['Train']==1) & (df['mileage']>1000)])
# посмотрим где проходят границы выбросов 

utils.borders_of_outliers('mileage',df[df['Train']==1], log = False)
# посмотрим какие это марки с такими пробегами более 432000

df[(df['Train']==1) & (df['mileage']>432000)].brand.value_counts()
# посмотрим статистику на тесте по пробегу

utils.describe_without_plots('mileage', df[df['Train']==0].mileage)
# сколько авто в тесте за границей

df[(df['Train']==0) & (df['mileage']>432000)].brand.value_counts()
# это полпроцента от объема теста удаляем все пробеги выше 432000 в трейне

df = df[((df['Train']==1) & (df['mileage']<=432000)) | ((df['Train']==0))]
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('mileage', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# напомню на этапе предварительного анализа мы не приводили этот признак к единому с тестом виду, поэтому надо будет полностью проанализировать значения и уже сразу понять в каком виде лучше их оставить для дальнейшего моделирования

# статистика по значениям в трейне

temp_df = df[df['Train']==1]

temp_df['Владельцы'].value_counts(normalize=True)
# статистика по значениям в тесте

temp_df = df[df['Train']==0]

temp_df['Владельцы'].value_counts(normalize=True)
# на этапе предварительного анализа были пропуски , проверим

len(df[df['Владельцы'].isna()])
temp_df['Владельцы'].unique()
# посмотрим статистику по пробегу по 2 владельцам 

df[(df['Владельцы']=='2\xa0владельца') & (df['mileage']>0)].mileage.describe()
# посмотрим статистику по пробегу по 3 и более владельцам 

df[(df['Владельцы']=='3 или более') & (df['mileage']>0)].mileage.describe()
# переведем в числовой формат 

# вытащим значение мощности перед первым пробелом'

df.loc[:, 'Владельцы'] = df['Владельцы'].apply(lambda x: int(x.split()[0]) if type(x)==str else None)
# заполним Владельцев по следующему принципу

# пробег от 100000 км - 2 владельца

# пробег от 150000 км - 3 владельца

# если меньше 100000 км - 1 владелец

# и тут нас ждет фиаско потому что пробег по этим автомобилям 0

df.loc[df['Владельцы'].isna()].mileage.describe()
# ну чтож посмотрим на пропуски в пробеге

df.loc[df['Владельцы'].isna()].head(3)
# вроде бы это новые автомобили, проверим

df.loc[df['Владельцы'].isna()].productionDate.describe()
# статистика по году производства в %

df[df['Владельцы'].isna()].productionDate.value_counts(normalize=True)
# видно что это новые автомобили, поэтому заполним Владельцев нулями

df['Владельцы'].fillna(0.0, inplace=True)
# переводим в int64

df['Владельцы'] = df['Владельцы'].astype('int64')
utils.describe_without_plots('Владельцы', df[df['Train']==1].Владельцы)
utils.four_plot_with_log2('Владельцы', df[df['Train']==1])
# фактически это категориальный признак с 4 значениями, но для корректной групповой обработки числовых признаков переведем его в числовые

cat_cols.remove('Владельцы')

num_cols.append('Владельцы')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('Владельцы', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# сначала посмотрим на кол-во пропусков 

df[df['Train']==1]['price'].isna().sum()
# затем как распределены пропуски по брендам

df[(df['Train']==1)&(df['price'].isna())]['brand'].value_counts()
# Пропуски цены есть практически во всех моделях. Удаляем эти данные, т.к. их менее 1%, и даже если мы их будем заполнять ближними значениями, тратить смысл на подгонку целевых переменных нет

df = df[((df['Train']==1)&(df['price'].isna()==False)) | (df['Train']==0)]
utils.describe_without_plots('price', df[df['Train']==1].price)
utils.four_plot_with_log2('price', df[df['Train']==1])
# распределение выглядит как логнормальное, посмотрим на выбросы и статистику мин-макс на BMW

# посмотрим где проходят границы выбросов 

utils.borders_of_outliers('price',df[df['Train']==1], log = True)
# а что по БМВ?

utils.describe_without_plots('price', df[(df['Train']==1) & (df['brand']=='BMW')].price)
# посмотрим сколько выбросов менее 60000 (мин БМВ) и более 	1.47812e+07 (макс БМВ)

len(df[(df['price']<60000) | (df['price']>1.47812e+07)])
# не много взглянем на них

df[(df['price']<60000) | (df['price']>1.47812e+07)].head(3)
# вроде как визуально старые авто проверим

df[(df['price']<60000) | (df['price']>1.47812e+07)].modelDate.value_counts(bins=5)
# нет не старые совсем даже

# убедимся что БМВ среди них нет и будем удалять

df[(df['price']<60000) | (df['price']>1.47812e+07)].brand.value_counts()
# удаляем выбросы по цене 

df = df[((df['Train']==1) & (df['price']>=60000) & (df['price']<1.47812e+07)) | (df['Train']==0)]
# еще раз смотрим на гистограммы

utils.four_plot_with_log2('price', df[df['Train']==1])
# ну теперь получилось очень красивое логнормальное распределение, хотя конечно необходимо было сначала проверить гипотезу о нормальном распределении, мы этот момент опустили из-за нехватки времени, просто создаем новый признак price_log логарифм от цены

df['price_log']=df['price'].apply(lambda x: np.log(x))



# добавим новый признак в список целевых, посмотрим что лучше обрабатывает медель потом при необходимости менее эффективный удалим

target_col.append('price_log')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('price', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
utils.describe_without_plots('modelDate', df[df['Train']==1].modelDate)
utils.describe_without_plots('modelDate', df[df['Train']==0].modelDate)
# пропусков нет

# в тесте нет моделей 2020 нет моделей младше 1975 года 

# посмотрим сколько их

len(df[(df['Train']==1) & ((df['modelDate']<1975) | (df['modelDate']>2019))])
# посмотрим на марки датой выпуска меньше 1975

df[(df['Train']==1) & (df['modelDate']<1975)].brand.value_counts()
# их немного 38 удаляем точно, а что с марками датой выпуска 2020

df[(df['Train']==1) & (df['modelDate']==2020)].brand.value_counts()
# можно было не смотреть марки 2020 года так как мы строим модель прогноза цен где таких моделей нет их нужно просто удалить по порогу и все

df = df[((df['Train']==1) & (df['modelDate']>=1975) & (df['modelDate']<=2019)) | (df['Train']==0)]
# посмотрим гистограммы

utils.four_plot_with_log2('modelDate', df[df['Train']==1])
# посмотрим на тест

utils.four_plot_with_log2('modelDate', df[df['Train']==0])
# визуально можно заметить что основная часть автомобилей, что в тестовой, что в тренировочной выборке начинается с начала-середины 90х

# посмотрим еще срезы по кол-ву значений чтоы убедиться в этом

# статистика по значениям modelDate в трейне

temp_df = df[df['Train']==1]

temp_df['modelDate'].value_counts(bins=10)
# статистика по значениям modelDate в тесте

temp_df = df[df['Train']==0]

temp_df['modelDate'].value_counts(bins=10)
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('modelDate', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# посмотрим на трейн

utils.describe_without_plots('productionDate', df[df['Train']==1].productionDate)
# теперь на тест

utils.describe_without_plots('productionDate', df[df['Train']==0].productionDate)
# посмотрим сколько авто с датой производства 2020

len(df[(df['Train']==1) & (df['productionDate']>2019)])
# такое большое кол-во это нормально так как датасет для соревнования формировался в феврале, то в тесте авто с датой производства 2020 нет - удалим их

# посмотрим сколько авто с датой производства менее 1981 в трейне

len(df[(df['Train']==1) & (df['productionDate']<1981)])
# удалим выбросы по порогу

df = df[((df['Train']==1) & (df['productionDate']>=1981) & (df['productionDate']<=2019)) | (df['Train']==0)]
utils.four_plot_with_log2('productionDate', df[df['Train']==1])
utils.four_plot_with_log2('productionDate', df[df['Train']==0])
# Посмотрим, как влияет год выпуска на распределение стоимости автомобиля

utils.four_plot_with_log2('price', df[(df['Train']==1)&(df['productionDate']<1990)])
utils.four_plot_with_log2('price', df[(df['Train']==1)&(df['productionDate']>1990)])
df[(df['Train']==1)&(df['productionDate']<1990)].price.describe()
# Посмотрим на авто, стоимость которых заметно выше остальных:

utils.four_plot_with_log2('price', df[(df['Train']==1)&(df['productionDate']<1990)&(df['price']>300000)])
# Добавим дополнительный признак 'intensity', который равен пробегу, поделенному на возраст авто, а также 'dateModelProdDiff', равный разнице между годом выпуска авто и годом начала производства модели

current_year = 2020

df['intensity']=df['mileage']/(current_year-df['productionDate'])

df['dateModelProdDiff']=df['productionDate']-df['modelDate']



# добавляем новые признаки

num_cols.append('intensity')

num_cols.append('dateModelProdDiff')
# Посмотрим на стоимость еще внимательнее

df[(df['Train']==1)&(df['productionDate']<1990)&(df['price']>1000000)].describe()
df[(df['Train']==1)&(df['productionDate']<1990)&(df['price']<1000000)].describe()
# Пока непонятно как выделять раритеты (~10% от датасета), предположим, что на это влияет интенсивность использования. 75% для раритетов это около 4215 км/год, а для не раритетов 25% начинается с 5670 км/год. Попробуем разделить по границе 5000 км/год, добавим дополнительный признак 'rarity'

df['rarity']=(df['intensity']<5000)&(df['productionDate']<1990)



# добавляем новые признаки

bin_cols.append('rarity')
# Посмотрим как меняется стоимость авто в зависимости от года выпуска

df_temp = df[(df['Train']==1)&(df['productionDate']<2020)&(df['rarity']==False)]

df_temp_bmw = df_temp[df['brand']=='BMW']

year = df_temp['productionDate'].values

price = df_temp['price'].values

year_bmw = df_temp_bmw['productionDate'].values

price_bmw = df_temp_bmw['price'].values

plt.figure(figsize=(20,10))

plt.scatter(year,price,c='b')

plt.scatter(year_bmw,price_bmw,c='r')
df_temp = df[(df['Train']==1)&(df['productionDate']<2005)&(df['rarity']==False)]

df_temp_bmw = df_temp[df['brand']=='BMW']

year = df_temp['productionDate'].values

price = df_temp['price'].values

year_bmw = df_temp_bmw['productionDate'].values

price_bmw = df_temp_bmw['price'].values

plt.figure(figsize=(20,10))

plt.scatter(year,price,c='b')

plt.scatter(year_bmw,price_bmw,c='r')
# добавляем 4 новых признака

df['pDate_more_2015']=df['productionDate']>=2015

df['pDate_more_2005']=(df['productionDate']>=2005)&(df['productionDate']<2015)

df['pDate_more_1990']=(df['productionDate']>=1990)&(df['productionDate']<2005)

df['pDate_less_1990']=(df['productionDate']<1990)



# добавляем новые признаки

bin_cols.append('pDate_more_2015')

bin_cols.append('pDate_more_2005')

bin_cols.append('pDate_more_1990')

bin_cols.append('pDate_less_1990')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('productionDate', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
temp_df = df[df['Train']==1]

temp_df['vehicleConfiguration'].value_counts(normalize=True)
# статистика по значениям в тесте

temp_df = df[df['Train']==0]

temp_df['vehicleConfiguration']
# удаляем

cat_cols.remove('vehicleConfiguration')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('vehicleConfiguration', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# статистика по значениям в трейне

temp_df = df[df['Train']==1]

temp_df['vehicleTransmission'].value_counts()
# статистика по значениям в тесте

temp_df = df[df['Train']==0]

temp_df['vehicleTransmission'].value_counts()
# удалим вариатор

df = df[((df['Train']==1) & (df['vehicleTransmission']!='вариатор')) | (df['Train']==0)]
# посмотрим на кол-во пропусков 

df[df['Train']==1]['vehicleTransmission'].isna().sum()
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('vehicleTransmission', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# можно вспомнить предварительный анализ трейна с помощью PandasProffiling из которого следовало что этот признак не заполнен даже на половину (это значение "{'id': '0'}" было доминирующим). Посмотрим что изменилось

temp_df = df[df['Train']==1]

(temp_df['Комплектация']=="{'id': '0'}").sum()/len(temp_df)
# более 75% процентов признака не заполнено в трейне, удаляем

cat_cols.remove('Комплектация')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('Комплектация', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# Описание в трейне

temp_df = df[df['Train']==1]

temp_df['description'].iloc[2]
# Описание в тесте

temp_df = df[df['Train']==0]

temp_df['description'].iloc[8]
# заполним пропуски

df['description'] = df['description'].fillna('[]')



# запишем списки слов в описании в отдельный столбец

df['words_in_description'] = df['description'].apply(lambda x: [str(i).lower() for i in x.split()])
# создаем новый признак кол-во слов в описании

df['count_words_d'] = df['description'].apply(lambda x: len(x.split()))



vectorizer = CountVectorizer()

text_feat = vectorizer.fit_transform(df['description'])



# создаем новые признаки кол-во среднее кол-во токенов и их сумма в описании

df['mean_c_w'] = text_feat.mean(axis=1)

df['sum_c_w'] = text_feat.sum(axis=1)



# удаляем 'description'

cat_cols.remove('description')



# добавляем 'count_words_d', 'mean_c_w', 'sum_c_w'

num_cols.append('count_words_d')

num_cols.append('mean_c_w')

num_cols.append('sum_c_w')
print(*df['words_in_description'][:3], sep='===')
# обработка слов в description

# выделяем словосочетания которые могут влиять на цену авто (дополнительный тюнинг или допопции при покупке нового авто) 

# защита картера  - crankcase protection

df['c_p_des1']= df['words_in_description'].apply(lambda x: 1 if ('защита' and 'картера') in x else 0)

bin_cols.append('c_p_des1')



# мультифункциональный руль - multifunction steering wheel

df['m_s_w_des2']= df['words_in_description'].apply(lambda x: 1 if ('мультифункциональный' and 'руль') in x else 0)

bin_cols.append('m_s_w_des2')



# датчики дождя и света - rain and light sensors

df['r_l_s_des3']= df['words_in_description'].apply(lambda x: 1 if ('датчики' and 'дождя' and 'света') in x else 0)

bin_cols.append('r_l_s_des3')



# АБС

df['abs_des4']= df['words_in_description'].apply(lambda x: 1 if ('антиблокировочная' and 'система') in x else 0)

bin_cols.append('abs_des4')



# круиз контроль - cruise control

df['c_c_des5']= df['words_in_description'].apply(lambda x: 1 if ('круиз-контроль') in x else 0)

bin_cols.append('c_c_des5')



# легкосплавные диски - alloy wheels

df['a_w_des6']= df['words_in_description'].apply(lambda x: 1 if ('легкосплавные' and 'диски') in x else 0)

bin_cols.append('a_w_des6')



# камера заднего вида - rear view camera

df['r_v_c_des7']= df['words_in_description'].apply(lambda x: 1 if ('камера' and 'видеокамера') in x else 0)

bin_cols.append('r_v_c_des7')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('description', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
temp_df = df[df['Train']==1]

temp_df['Привод'].value_counts()
# статистика по значениям в тесте

temp_df = df[df['Train']==0]

temp_df['Привод'].value_counts()
# посмотрим на кол-во пропусков 

df[df['Train']==1]['Привод'].isna().sum()
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('Привод', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
temp_df = df[df['Train']==0]

temp_df['Состояние'].value_counts()
# удалим Состояние потому что не понятно как его обработать

cat_cols.remove('Состояние')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('Состояние', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
temp_df = df[df['Train']==1]

temp_df['ПТС'].value_counts()
temp_df['ПТС'].isna().sum()
temp_df[temp_df['ПТС'].isna()].head(3)
# статистика по значениям в тесте

temp_df = df[df['Train']==0]

temp_df['ПТС'].value_counts()
# Заполним отсутствующие значения вариантом "нет", т.к. это новые авто без ПТС

df['ПТС'].fillna('Нет', inplace = True)
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('ПТС', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# Значения в трейне

temp_df = df[df['Train']==1]

temp_df['Владение'].value_counts()
# Значения в тесте

temp_df = df[df['Train']==0]

temp_df['Владение']
#заполняем пропуски значением nodata

df['Владение'] = df['Владение'].fillna('nodata')



def months_to_sent(months):

    if months == 1:

        return f'{months} месяц'

    elif 2 <= months <= 4:

        return f'{months} месяца'

    return f'{months} месяцев'

def years_to_sent(years):

    if 11 <= years <= 14 or 5 <= years%10 <= 9 or years%10 == 0:

        return f'{years} лет'

    elif years%10 == 1:

        return f'{years} год'

    elif 2 <= years%10 <= 4:

        return f'{years} годa'

def tenure(row):

    row = re.findall('\d+',row)

    if row != []:

        years = 2020 - (int(row[0])+1)

        months = 2 +(12 - int(row[1]))

        if years < 0:

            return months_to_sent(int(row[1]))

        elif years == 0 and months < 12:

            return months_to_sent(months)

        elif years >= 0 and months == 12:

            return years_to_sent(years + 1)

        elif years >= 0 and months > 12:

            return years_to_sent(years + 1)+' и '+months_to_sent(months - 12)

        elif years > 0 and months < 12:

            return years_to_sent(years)+' и '+months_to_sent(months)

        return None

    

df.loc[df['Train']==1,'Владение'] = df[df['Train']==1]['Владение'].apply(tenure)
def num_of_months(row):

    if pd.notnull(row) and row!='nodata':

        list_ownership = row.split()

        if len(list_ownership) == 2:

            if list_ownership[1] in ['год', 'года', 'лет']:

                return int(list_ownership[0])*12

            return int(list_ownership[0])

        return int(list_ownership[0])*12 + int(list_ownership[3])
df['num_of_month'] = df['Владение'].apply(num_of_months)

# добавляем кол-во месяцев владения

num_cols.append('num_of_month')

# удаляем владение 

cat_cols.remove('Владение')
df['num_of_month'].value_counts(bins=50, normalize=True)[:10]
df['num_of_month'].describe()
df[df['num_of_month']>0].describe()
# колво пропусков месяцев с нулевым пробегом

len(df[df['num_of_month'].isna() & (df['mileage']==0)])
df[df['num_of_month']<3].head(3)
df[(df['num_of_month']>0)&(df['num_of_month']<100)]['num_of_month'].hist(bins=100)
# надо удалить владение, так как перед тем как над ним применять какиенибудь преобразования надо его заполнить, я заполнить его не представляется возможным так как много пропусков

num_cols.remove('num_of_month')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('Владение', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# Значения в трейне

temp_df = df[df['Train']==1]

temp_df['name'].value_counts()
# Значения в тесте

temp_df = df[df['Train']==0]

temp_df['name']
# добавим что-то напоминающее модель в категориальные признаки 

df['model2'] = df['name'].apply(lambda x: str(x).split()[0])

cat_cols.append('model2')

# добавим что-то напоминающее характеристику модели в категориальные признаки 

df['model2_2'] = df['name'].apply(lambda x: str(x).split()[1])

cat_cols.append('model2_2')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('name', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# заполним значения модели по тесту , так как модель определяет класс автомобиля. А класс определяет цену. Сделаем это с помощью CatBoost



# приведем числовые критерии к int64 для CatBoost

for col in ['modelDate', 'productionDate', 'engineDisplacement2']:

    df[col] = df[col].astype('int64')
# позволяет закрепить random_state во всей ячейке исполнения

np.random.seed(42)



le = LabelEncoder()



le.fit(df['bodyType'])

df['bodyType_'] = le.transform(df['bodyType'])



le.fit(df['name'])

df['name_'] = le.transform(df['name'])



le.fit(df['model2'])

df['model2_'] = le.transform(df['model2'])



le.fit(df['model2_2'])

df['model2_2_'] = le.transform(df['model2_2'])



le.fit(df['Привод'])

df['Привод_'] = le.transform(df['Привод'])



le.fit(df['fuelType'])

df['fuelType_'] = le.transform(df['fuelType'])



le.fit(df['vehicleTransmission'])

df['vehicleTransmission_'] = le.transform(df['vehicleTransmission'])



le.fit( df[df['Train']==1]['model'])

df.loc[df['Train']==1, 'model_'] = le.transform(df[df['Train']==1]['model'])
# выведем соответсвия номеров в model_ и моделей БМВ

temp_train = df[(df['Train']==1) & (df['brand']=='BMW')]

list_model = list(temp_train.model_.unique())

for i in list_model:

    print(i, '==', temp_train[temp_train['model_']==i].model.values[0])
# создаем список признаков для CatBoost

cols_for_CatBoost = ['bodyType_', 'name_', 'model2_', 'model2_2_', 'modelDate', 'engineDisplacement2', 'Привод_', 'fuelType_', 'enginePower', 'vehicleTransmission_']
temp_train = df[(df['Train']==1) & (df['brand']=='BMW')]

temp_test = df[(df['Train']==0)]

train_data = temp_train[cols_for_CatBoost] # обучающая выборка

train_labels = temp_train['model_'] # метки принадлежности к классу

test_data = temp_test[cols_for_CatBoost] # тестовая выборка



model = CatBoostClassifier(iterations=50, learning_rate = 0.5, random_state=RANDOM_SEED) # классификатор

model.fit(train_data, train_labels) # обучение классификатора

prediction = model.predict(test_data) # передача тестовой выборки в модель

print(*list(prediction[:10])) # вывод результата "предсказания"
# проверим первые 10 элементов теста

temp_test[['brand','bodyType', 'name', 'model', 'model2', 'modelDate', 'engineDisplacement2', 'Привод', 'fuelType', 'enginePower', 'vehicleTransmission']][:10]
# выведем соотстсвующие номера после проверки

print(*[20, 13, 20, 416, 414, 416, 412, 410, 412, 20])
print(*list(prediction[:10])) # вывод результата "предсказания"
df.loc[df['Train']==0, 'model_']=prediction
# больше name нам не нужно удалим его

cat_cols.remove('name')

# также удалим model потому что теперьу нас есть числовой признак модели model_

cat_cols.remove('model')

# добавим новый признак

cat_cols.append('model_')
# сохраним список номеров всех моделей BMW

list_all_num_model_BMW = df[df['brand']=='BMW'].model_.unique()
# проверим пропуски

df['model_'].isna().sum()
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('model', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# удалим start_date потому что не понятно как его обработать

time_cols.remove('start_date')
# записываем признак в список проанализированных признаков

old_len_train, EDA_done_cols = utils.result_EDA_feature('start_date', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
# Значения в трейне

temp_df = df[df['Train']==1]

temp_df['color'].value_counts()
# Значения в трейне

temp_df = df[df['Train']==0]

temp_df['color'].value_counts()
# записываем признак в список проанализированных признаков

EDA_done_cols.append('color')

old_len_train, EDA_done_cols = utils.result_EDA_feature('', df[df['Train']==1], df[df['Train']==0], 23, EDA_done_cols, old_len_train)
#  этот раздел удален так как на каггле не требуется, оставлен только переход к df3



# сохраняем все переменные из списков

all_cols_df3 =cat_cols+num_cols+time_cols+servis_cols+bin_cols+target_col

# образаем исходный датасет только переменными, которые мы решили оставить

df3 = df.loc[:, all_cols_df3].copy()

utils.describe_without_plots_all_collumns(df3, short=True)
utils.simple_heatmap('Матрица корреляции тренировочного датасета на числовых переменных',df3[df3['Train']==1], num_cols+target_col, 1.1, 1, 9)
temp_df = df3[(df3['Train']==1) & (df3['brand']=='BMW')]

utils.simple_heatmap('Матрица корреляции тренировочного датасета на числовых переменных по  BMW',temp_df, num_cols+target_col, 1.1, 1, 9)
temp_df = df3[df3['Train']==1]

imp_num = pd.Series(f_classif(temp_df[num_cols], temp_df['price_log'])[0], index = num_cols)

imp_num.sort_values(inplace = True)

imp_num.plot(kind = 'barh', title='Значимость непрерывных переменных по ANOVA F test по всем маркам')
temp_df = df3[(df3['Train']==1) & (df3['brand']=='BMW')]

imp_num = pd.Series(f_classif(temp_df[num_cols], temp_df['price_log'])[0], index = num_cols)

imp_num.sort_values(inplace = True)

imp_num.plot(kind = 'barh', title='Значимость непрерывных переменных по ANOVA F test по BMW')
# переведем категориальные признаки в dummies переменные

# но сначала сохраним список переменных чтобы можно было сделать список добавленных

list_cols_bef_dumm = list(df3.columns)



# а также мы хотим преобразовать числовые признаки 'Владельцы' и 'engineDisplacement2'

# поэтому сохраним их дубликаты

arr_Владельцы = np.array(df3['Владельцы'])



# преобразуем переменные в дамми переменные

df3 = pd.get_dummies(df3, columns = ['bodyType', 'brand', 'color', 'fuelType', 'vehicleTransmission', 'Привод', 'Владельцы', 'ПТС', 'model_', 'engineDisplacement2'])



# вернем владельцев

df3['Владельцы'] = arr_Владельцы 



# теперь создадим список дамми переменных

list_cols_aft_dumm = list(df3.columns)

dumm_cols= list(set(list_cols_aft_dumm)-set(list_cols_bef_dumm ))
print(f'Мы добавили:{len(dumm_cols)} dummies featuries')
# удалим 'engineDisplacement2' из числовых признаков - они нам больше не понадобятся

num_cols.remove('engineDisplacement2')



# а также удалим категориальные признаки преобразованные в dummies их уже нет

drop_list_cols = ['bodyType', 'brand', 'color', 'fuelType', 'vehicleTransmission', 'Привод', 'ПТС', 'model_']

for col in drop_list_cols:

    cat_cols.remove(col)
# Проверим, есть ли статистическая разница в распределении оценок по всем категориальным признакам, 

# с помощью теста Стьюдента. Проверим нулевую гипотезу о том, 

# что распределения price_log по различным параметрам неразличимы:

def get_stat_dif(d_column, d_df):

    cols = d_df.loc[:, d_column].value_counts().index[:]

    combinations_all = list(combinations(cols, 2))

    for comb in combinations_all:

        ttest = ttest_ind(d_df.loc[d_df.loc[:, d_column] == comb[0], 'price_log'].dropna(),

                          d_df.loc[d_df.loc[:, d_column] == comb[1], 'price_log'].dropna()).pvalue

        if  ttest<= 0.05/len(combinations_all): # Учли поправку Бонферони

            return(d_column)

            break
stat_sign_diff_cols=[]



temp_df = df3[(df3['Train']==1)]

for col in bin_cols+dumm_cols:

    stat_sign_diff_cols.append(get_stat_dif(col,temp_df))

stat_sign_diff_cols = list(filter(None, stat_sign_diff_cols))
not_stat_sign_diff_cols=list(set(bin_cols+dumm_cols) - set(stat_sign_diff_cols))

print(f'по тесту Стьюдента {len(not_stat_sign_diff_cols)} признаков из {len(bin_cols+dumm_cols)} бинарных и dummies признаков НЕ СОДЕРЖАТ СТАТИСТИЧЕСКИ ЗНАЧИМЫХ РАЗЛИЧИЙ с таргетом')

print('вот они:= ', *sorted(not_stat_sign_diff_cols))
dumm_cols.remove('brand_MINI')

df3 = df3[df3['brand_MINI']!=1]
check_list_model = [x for x in sorted(not_stat_sign_diff_cols) if 'model' in x]

print(f'всего моделей на проверку на удаление:= {len(check_list_model)}')
# ранее мы сохранили список всех номеров моделей БМВ

print(*list_all_num_model_BMW)
check_list_model_BMW = ['model__'+ str(x) for x in list_all_num_model_BMW]

print(*check_list_model_BMW)
fin_drop_list_model = list(set(check_list_model)-set(check_list_model_BMW))

print(f'всего удаляем моделей после проверки:= {len(fin_drop_list_model)}')
for col in fin_drop_list_model:

    dumm_cols.remove(col)

    df3 = df3[df3[col]!=1]
# сколько в трейне

len(df3[df3['Train']==1])
# посмотрим сколько осталось авто БМВ в трейне

len(df3[(df3['Train']==1) & (df3['brand_BMW']==1)])
# сколько в тесте

len(df3[df3['Train']==0])
# псомотрим еще раз на распределение авто бмв в тесте по дате марки

df3[(df3['Train']==0) & (df3['brand_BMW']==1)].productionDate.value_counts(bins=10, normalize=True)
# Сделаем валидационную выборку из 800 авто

temp_ser = df3[(df3['Train']==0) & (df3['brand_BMW']==1)].productionDate.value_counts(bins=10, normalize=True)

temp_list = []

for interval in list(temp_ser.index):

    temp_list.append([interval, int(temp_ser[interval]*800)])

temp_list
# пронумеруем трейн чтобы убрать из трейна то что попаедет в валидационнную выборку

df3.loc[df3['Train']==1,'id'] = np.array(range(1000000, 1000000+len(df3[df3['Train']==1])))
# позволяет закрепить random_state во всей ячейке исполнения

np.random.seed(42)



df_val = pd.DataFrame()

i = 0

dict_id_val = {}

for interval in temp_list:

    df_sample = df3[(df3['Train']==1) & (df3['brand_BMW']==1) & (df3['productionDate'] > interval[0].left) & (df3['productionDate'] <= interval[0].right)].sample(interval[1]).copy()

    dict_id_val[i]=list(df_sample.id.unique())

    df_val = pd.concat([df_val, df_sample])

    i += 1

len(df_val)
# визуальный контроль

df_val.head(2)
df_train = df3[df3['Train']==1].copy()

# сделаем трейн без авто в валидационной выборке

for elem in dict_id_val:

    df_train = df_train[~df_train['id'].isin(dict_id_val[elem])]

len(df_train)
# проверка сумма длин трайна и валидации = первоначальному трейну

len(df_train)+len(df_val) == len(df3[df3['Train']==1])
# проверим распределение по типу топлива, чтобы убедиться что у нас в валидации доставточно гибридов и электрокаров

dumm_fuelType_cols = [x for x in dumm_cols if 'fuelType' in x]



print('Распределение fuelType в валидации:')

for col in dumm_fuelType_cols:

    print(col, ':=', int(len(df_val[df_val[col]==1])/len(df_val)*10000)/100, '%')
# проверим распределение по типу топлива, чтобы убедиться что у нас в валидации доставточно гибридов и электрокаров

dumm_fuelType_cols = [x for x in dumm_cols if 'fuelType' in x]

temp_df = df3[df3['Train']==0]

print('Распределение fuelType в тесте:')

for col in dumm_fuelType_cols:

    print(col, ':=', int(len(temp_df[temp_df[col]==1])/len(temp_df)*100000)/1000, '%')
# поммотрим какие электрокары в валидации

df_val[df_val['fuelType_электро']==1]
# удаляем электрокар 2017 года так как этот интервал шире в тесте (см.выше)

# сначала добавляем в трейн

df_train = pd.concat([df_train,df_val[df_val['id']==1004755.0]])



# потом удаляем в валидации

df_val = df_val[df_val['id']!=1004755.0]
# позволяет закрепить random_state во всей ячейке исполнения

np.random.seed(42)



# добавляем в валидацию 5 гибридов БМВ

df_sample = df_train[(df_train['brand_BMW']==1) & (df_train['fuelType_гибрид']==1)].sample(5).copy()

df_val = pd.concat([df_val, df_sample])



df_train = df_train[~df_train['id'].isin(list(df_sample.id.unique()))]
# проверка сумма длин трайна и валидации = первоначальному трейну

len(df_train)+len(df_val) == len(df3[df3['Train']==1])
# сохраняем все переменные из списков которые находятся в работе кроме дамми

all_cols_df4 =cat_cols+num_cols+servis_cols+bin_cols+target_col

# образаем исходный датасет только переменными, которые мы решили оставить

df4 = df_train.loc[:, all_cols_df4].copy()

utils.describe_without_plots_all_collumns(df4, short=True)
df_train.head(2)
df_val.head(2)
w_n = np.random.normal(0, 1, size=len(df_train)+len(df_val))

df_train['w_n']=w_n[:len(df_train)]

df_val['w_n']=w_n[-len(df_val):]
train = df_train.drop(['model2', 'model2_2', 'Train', 'id', 'price'], axis =1)

val = df_val.drop(['model2', 'model2_2', 'Train', 'id', 'price'], axis =1)



y_train = train.price_log.values            # наш таргет

X_train = train.drop(['price_log'], axis=1)



y_val = val.price_log.values            # наш таргет

X_val = val.drop(['price_log'], axis=1)



# проверяем

train.shape, X_train.shape, y_train.shape, val.shape, X_val.shape, y_val.shape
model = RandomForestRegressor(random_state = RANDOM_SEED, n_jobs = -1, verbose = 1)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
# в первый раз инициируем глобальную переменную с предыдущим скором

utils.last_pred = np.zeros((3,len(y_val)))
utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), np.exp(y_pred))
plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)

feat_importances.nlargest(15).plot(kind='barh')
# попробуем Случайный лес только на БМВ

train = df_train[df_train['brand_BMW']==1].drop(['model2', 'model2_2', 'Train', 'id', 'price'], axis =1)



y_train = train.price_log.values            # наш таргет

X_train = train.drop(['price_log'], axis=1)



# проверяем

train.shape, X_train.shape, y_train.shape, val.shape, X_val.shape, y_val.shape
model = RandomForestRegressor(random_state = RANDOM_SEED, n_jobs = -1, verbose = 1)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), np.exp(y_pred))
# теперь Линейную регрессию на всем трейне

# Проверим LinearRegression

train = df_train.drop(['model2', 'model2_2', 'Train', 'id', 'price'], axis =1)

val = df_val.drop(['model2', 'model2_2', 'Train', 'id', 'price'], axis =1)



y_train = train.price_log.values            # наш таргет

X_train = train.drop(['price_log'], axis=1)



lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), np.exp(y_pred))
# попробуем ExtraTreeRegressor на всем трейне

etr = ExtraTreeRegressor(random_state = RANDOM_SEED)

etr.fit(X_train, y_train)

y_pred = etr.predict(X_val)

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), np.exp(y_pred))
# проверим градиентный бустинг на всем трейне

gbr = GradientBoostingRegressor(n_estimators=250)

gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_val)

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), np.exp(y_pred))
# проверим BaggingRegressor вместе со случайным лесом на всем трейне

bgr_rf = BaggingRegressor(model, n_estimators=3, n_jobs=-1, random_state=RANDOM_SEED)

bgr_rf.fit(X_train, y_train)

y_pred = bgr_rf.predict(X_val)

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), np.exp(y_pred))
model = RandomForestRegressor(random_state = RANDOM_SEED, n_jobs = -1, verbose = 1)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), np.exp(y_pred))
# попробуем стекинг Случайного леса и беггинг, пока без мета модели, просто возьмем  среднее

models = [RandomForestRegressor(random_state = RANDOM_SEED, n_jobs = -1, verbose = 1),

         BaggingRegressor(ExtraTreeRegressor(random_state=RANDOM_SEED), random_state=RANDOM_SEED)]



def stacking_model_predict(d_models, d_X_train, d_y_train, d_X_val):

    d_df = pd.DataFrame()

    for model_ in tqdm(d_models):

        model_.fit(d_X_train, d_y_train)

        y_pred = model_.predict(d_X_val)

        d_df[str(model_)[:6]] = np.exp(y_pred)                   

    return d_df



temp_df = stacking_model_predict(models, X_train, y_train, X_val)

temp_df['y_pred']=temp_df.mean(axis=1)
y_pred = np.array(temp_df['y_pred'])

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), y_pred)
y_pred = np.round(np.array(temp_df['y_pred'])/1000,2)*1000

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), y_pred)
y_pred = np.round(np.array(temp_df['y_pred'])/100,2)*100

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), y_pred)
y_pred = np.round(np.array(temp_df['y_pred'])/10000,2)*10000

utils.test_last_pred(y_val, y_pred, y_pred) if (utils.last_pred[0].max() == 0) else 0

utils.all_metrics_MAE_MPE_MAPE_WAPE_MSE_RMSE(np.exp(y_val), y_pred)
y_pred = np.round(np.array(temp_df['y_pred'])/10000,2)*10000

df_val['MAPE'] = np.round(np.abs((np.exp(y_val) - y_pred)/np.exp(y_val))*100,4)

df_val['y_pred'] = y_pred
df_val['MAPE'].describe()
df_val[df_val['MAPE']>35][['price', 'y_pred', 'MAPE', 'model2', 'modelDate','productionDate', 'enginePower', 'intensity', 'rarity', 'mileage', 'fuelType_бензин']]
df_val[df_val['MAPE']>35][['price', 'y_pred', 'MAPE', 'model2', 'modelDate','productionDate', 'enginePower', 'intensity', 'rarity', 'mileage', 'fuelType_бензин']].describe()
# # закомментирован так как выполняется очень долго

# # позволяет закрепить random_state во всей ячейке исполнения

# np.random.seed(42)



# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1500, num = 50)]

# max_features = ['auto', 'sqrt', 'log2']

# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

# max_depth.append(None)

# min_samples_split = [2, 5, 10]

# min_samples_leaf = [1, 2, 4]

# bootstrap = [True, False]

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}



# rf = RandomForestRegressor()

# rf_random = RandomizedSearchCV(estimator = rf, 

#                                param_distributions = random_grid, 

#                                n_iter = 100, 

#                                cv = 3, 

#                                verbose=2, 

#                                random_state=RANDOM_SEED, 

#                                n_jobs = -1)

# rf_random.fit(X_train, y_train)
train = df3.query('Train==1').drop(['model2', 'model2_2', 'Train', 'id', 'price'], axis = 1)

test = df3.query('Train==0').drop(['model2', 'model2_2', 'Train', 'id', 'price'], axis = 1)
y_train = train.price_log.values            # наш таргет

X_train = train.drop(['price_log'], axis=1)



X_test = test.drop(['price_log'], axis=1)
models = [RandomForestRegressor(random_state = RANDOM_SEED, n_jobs = -1, verbose = 1),

         BaggingRegressor(ExtraTreeRegressor(random_state=RANDOM_SEED), random_state=RANDOM_SEED)]



def stacking_model_predict(d_models, d_X_train, d_y_train, d_X_test, d_df):

    for model_ in tqdm(d_models):

        model_.fit(d_X_train, d_y_train)

        y_pred = model_.predict(d_X_test)

        d_df[str(model_)[:6]] = np.round(np.exp(y_pred)/10000,2)*10000

    d_df['price'] = d_df.iloc[:,2:].mean(axis=1)    

    return 



stacking_model_predict(models, X_train, y_train, X_test, df_submit)
df_submit.head(5)
df_submit[['id', 'price']].to_csv(f'submission.csv', index=False)

