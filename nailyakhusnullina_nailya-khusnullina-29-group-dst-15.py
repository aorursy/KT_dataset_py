# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import datetime

import re

import random



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



# Feature scaling with StandardScaler 

from sklearn.preprocessing import StandardScaler 



# Feature scaling with MinMaxScaler 

from sklearn.preprocessing import MinMaxScaler 



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



data = df_train.append(df_test, sort=False).reset_index(drop=True) # объединяем
data.info()
data.Restaurant_id
data.Restaurant_id = data.Restaurant_id.apply(lambda x: float(x[3:]))
len(data[data.Restaurant_id == 100])
data.Restaurant_id.hist(bins=100)
# рассчитаем границы выбросов, если таковые имеются



#median = data.Restaurant_id.median()

#IQR = data.Restaurant_id.quantile(0.75) - data.Restaurant_id.quantile(0.25)

#perc25 = data.Restaurant_id.quantile(0.25)

#perc75 = data.Restaurant_id.quantile(0.75)

#print('25-й перцентиль: {},'.format(perc25),

#      '75-й перцентиль: {},'.format(perc75),

#      "IQR: {}, ".format(IQR),

#      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))



#data.Restaurant_id.loc[data.Restaurant_id.between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins = 200, 

#                                                                        range = (0, 101),

#                                                                        color = 'blue',

#                                                                        label = 'IQR')



#plt.legend();
len(data.City.value_counts())
data.City.value_counts()
data['City'].value_counts(ascending=True).plot(kind='barh')
# Признак is_capital - является ли город столицей



#data['is_capital'] = data.apply(lambda x: 1 if x.City == 'Paris' or x.City == 'Stockholm' or x.City == 'London' 

#                            or x.City == 'Berlin' or x.City == 'Bratislava' or x.City == 'Vienna' 

#                            or x.City == 'Rome' or x.City == 'Madrid' or x.City == 'Dublin' 

#                            or x.City == 'Brussels' or x.City == 'Warsaw' or x.City == 'Budapest'

#                            or x.City == 'Copenhagen' or x.City == 'Amsterdam' or x.City == 'Lisbon' 

#                            or x.City == 'Prague' or x.City == 'Oslo' or x.City == 'Helsinki'

#                            or x.City == 'Ljubljana' or x.City == 'Athens' or x.City == 'Luxembourg' 

#                            else 0, axis=1)
# dummy-параметры на основании столбца City



dummies = pd.get_dummies(data['City'])

data = pd.concat([data, dummies], axis=1)
#data.sample(5)
# Признак City_Life_Quality - рейтинг качества жизни в городах



data['City_Life_Quality'] = data.apply(lambda x: 39 if x.City == 'Paris' else 23 if x.City == 'Stockholm' 

                                   else 41 if x.City == 'London' else 13 if x.City == 'Berlin' 

                                   else 38 if x.City == 'Oporto' else 41 if x.City == 'Milan'

                                   else 80 if x.City == 'Bratislava' else 56 if x.City == 'Rome'

                                   else 43 if x.City == 'Barcelona'else 46 if x.City == 'Madrid'

                                   else 33 if x.City == 'Dublin' else 28 if x.City == 'Brussels'

                                   else 82 if x.City == 'Warsaw' else 76 if x.City == 'Budapest'

                                   else 8 if x.City == 'Copenhagen' else 11 if x.City == 'Amsterdam'

                                   else 40 if x.City == 'Lyon' else 19 if x.City == 'Hamburg'

                                   else 37 if x.City == 'Lisbon' else 69 if x.City == 'Prague'

                                   else 25 if x.City == 'Oslo' else 31 if x.City == 'Helsinki'

                                   else 45 if x.City == 'Edinburgh' else 9 if x.City == 'Geneva'

                                   else 74 if x.City == 'Ljubljana' else 89 if x.City == 'Athens'

                                   else 18 if x.City == 'Luxembourg' else 100 if x.City == 'Krakow'

                                   else 1 if x.City == 'Vienna' else 2 if x.City == 'Zurich' 

                                   else 3 if x.City == 'Munich' else 1000, axis=1)



# https://mobilityexchange.mercer.com/Insights/quality-of-living-rankings
# Признак City_Population - количество жителей в городах 



#data['City_Population'] = data.apply(lambda x: 11400000 if x.City == 'Paris' else 2225000 if x.City == 'Stockholm' 

#                                   else 14800000 if x.City == 'London' else 4725000 if x.City == 'Berlin' 

#                                   else 1540000 if x.City == 'Oporto' else 6200000 if x.City == 'Milan'

#                                   else 4600000 if x.City == 'Bratislava' else 3575000 if x.City == 'Rome'

#                                   else 4775000 if x.City == 'Barcelona'else 6550000 if x.City == 'Madrid'

#                                   else 1390000 if x.City == 'Dublin' else 2700000 if x.City == 'Brussels'

#                                   else 2375000 if x.City == 'Warsaw' else 2625000 if x.City == 'Budapest'

#                                   else 1680000 if x.City == 'Copenhagen' else 2475000 if x.City == 'Amsterdam'

#                                   else 1980000 if x.City == 'Lyon' else 2875000 if x.City == 'Hamburg'

#                                   else 2475000 if x.City == 'Lisbon' else 1440000 if x.City == 'Prague'

#                                   else 1190000 if x.City == 'Oslo' else 1300000 if x.City == 'Helsinki'

#                                   else 537000 if x.City == 'Edinburgh' else 614000 if x.City == 'Geneva'

#                                   else 288000 if x.City == 'Ljubljana' else 3350000 if x.City == 'Athens'

#                                   else 626000 if x.City == 'Luxembourg' else 1000000 if x.City == 'Krakow'

#                                   else 2250000 if x.City == 'Vienna' else 1430000 if x.City == 'Zurich' 

#                                   else 2250000 if x.City == 'Munich' else 1000, axis=1)



# https://en.wikipedia.org/wiki/List_of_metropolitan_areas_in_Europe
data['Cuisine Style']
data['Nan_Cuisine_Style'] = pd.isna(data['Cuisine Style']).astype('uint8')
data['Cuisine Style'].loc[0]
# Признак Num_Cuisine - количесво кухонь, представленных в ресторане



data['Num_Cuisine'] = data['Cuisine Style'].str.count(',') + 1

data['Num_Cuisine'].fillna(1, inplace=True)
(pd.Series(data['Cuisine Style'].str.cat(sep=',').replace("[", '').replace("]","").replace(' ', '').replace("'", '').split(',')).value_counts())
data['Cuisine Style'].fillna("['European', 'Vegetarian Friendly']", inplace=True)
# Признак Country - страна 



data['Country'] = data.apply(lambda x: 'France' if x.City == 'Paris' or x.City == 'Lyon'

                         else 'Sweden' if x.City == 'Stockholm' else 'UK' if x.City == 'London' 

                         or x.City == 'Edinburgh' else 'Germany' if x.City == 'Berlin' or x.City == 'Munich' 

                         or x.City == 'Hamburg' else 'Portugal' if x.City == 'Oporto' or x.City == 'Lisbon'

                         else 'Italy' if x.City == 'Milan' or x.City == 'Rome' else 'Slovakia' if x.City == 'Bratislava'

                         else 'Austria' if x.City == 'Vienna' else 'Spain' if x.City == 'Barcelona' or x.City == 'Madrid'

                         else 'Ireland' if x.City == 'Dublin' else 'Belgium' if x.City == 'Brussels'

                         else 'Switzerland' if x.City == 'Zurich' or x.City == 'Geneva' 

                         else 'Poland' if x.City == 'Warsaw' or x.City == 'Krakow'

                         else 'Hungary' if x.City == 'Budapest' else 'Denmark' if x.City == 'Copenhagen' 

                         else 'Netherlands' if x.City == 'Amsterdam' else 'Czech Republic' if x.City == 'Prague'

                         else 'Norway' if x.City == 'Oslo' else 'Finland' if x.City == 'Helsinki'

                         else 'Slovenia' if x.City == 'Ljubljana' else 'Greece' if x.City == 'Athens'

                         else 'Luxembourg' if x.City == 'Luxembourg' else 1000, axis=1)
len(data['Country'].value_counts())
# Признак local_cuisine - наличие локальной кухни в Cuisine Style



data['local_cuisine'] = 0

 

# введем два списка - страны, которые есть в датасете, и соответствующие странам названия кухонь



list_of_Countries = ['France', 'Sweden', 'UK', 'Germany', 'Portugal', 'Italy', 'Slovakia', 'Austria',

                     'Spain', 'Ireland', 'Belgium', 'Switzerland', 'Poland', 'Hungary', 'Denmark', 

                     'Netherlands', 'Czech Republic', 'Norway', 'Finland', 'Slovenia', 'Greece',

                     'Luxembourg']



list_of_Country_Cuisines = ['French', 'Swedish', 'British', 'German', 'Portuguese', 'Italian', 'Slovak',

                    'Austrian', 'Spanish', 'Irish', 'Belgian', 'Swiss', 'Polish', 'Hungarian', 'Danish',

                    'Dutch', 'Czech', 'Norwegian', 'Finnish', 'Slovenian', 'Greek', 'Luxembourg']



# Для каждой локальной кухни страны проверяем условие наличия этой кухни в Cuisine Style и страну 

# (например, французская кухня в Италии не будет локальной, а во Франции - да)



for i in range(len(list_of_Country_Cuisines)):

    data.loc[(data['Cuisine Style'].str.contains(list_of_Country_Cuisines[i])) & (data.Country == list_of_Countries[i]), 'local_cuisine'] = 1
# Признак vegeterian_friendly - наличие в Cuisine Style вегетарианского или веганского меню



data['vegeterian_friendly'] = 0

data.loc[data['Cuisine Style'].str.contains('Vegetarian') == True, 'vegeterian_friendly'] = 1

data.loc[data['Cuisine Style'].str.contains('Vegan') == True, 'vegeterian_friendly'] = 1
# Признак pop_cuisine - наличие в Cuisine Style европейского (European) меню



#data['pop_cuisine'] = 0

#data.loc[data['Cuisine Style'].str.contains('European') == True, 'pop_cuisine'] = 1
# dummy-параметры на основании столбца Cuisine Style



Cuisine_Style_List = list(set(data['Cuisine Style'].str.cat(sep=',').replace("[", '').replace("]","").replace("'", '').replace(' ', '').split(',')))

# 'Vegetarian', 'Vegan', 'European' исключены, тк они вынесены отдельно как популярные и Vegeterian Friendly

Cuisine_Style_List = [a for a in Cuisine_Style_List if a != 'VegetarianFriendly' and a != 'VeganOptions' and a != 'European']



for i in range(len(Cuisine_Style_List)):

    data[Cuisine_Style_List[i]] = 0

    data.loc[data['Cuisine Style'].str.contains(Cuisine_Style_List[i], case=False, na=False), Cuisine_Style_List[i]] = 1
city_list = list(data.City.unique())
# Признак Pizza_Num - в каком городе больше всего пиццерий



for city in city_list:

    data.loc[data.City == city, 'Pizza_Num'] = data.loc[data.City == city, 'Pizza'].sum()
# Признак restaurant_num_in_City - количество ресторанов в городе



#data['restaurant_num_in_City'] = 0

for city in city_list:

    data.loc[data.City == city, 'restaurant_num_in_City'] = data[data.City == city].City.count()
# dummy признак для Country



#dummies = pd.get_dummies(data['Country'])

#data = pd.concat([data, dummies], axis=1)
data.Ranking.sample(5)
data.Ranking.hist(bins=100)
data['Ranking'][data['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (data['City'].value_counts())[0:10].index:

    data['Ranking'][data['City'] == x].hist(bins=100)

plt.show()
list_of_City = data.City.unique()
# Стандартная нормировка Restaurant_id и Ranking 

# (x - x.mean)/Standart_Deviation



#data['std_id'] = 0 

#data['std_Ranking'] = 0

##data = data.sort_values(by = ['City', 'Ranking'], axis=0).reset_index(drop=True)

#for City in list_of_City:

#    loc_data = data[data.City == City]

#    scale_features_std = StandardScaler() 

#    # Берём Restaurant_id потому, что StandardScaler() требует два значения

#    std_id_Ranking = scale_features_std.fit_transform(loc_data[['Restaurant_id','Ranking']]) 

#    j=0

#    for i in data.index[data['City']==City]:

#        data.loc[i, 'std_id']= pd.DataFrame(std_id_Ranking[:,0]).loc[j][0]

#        data.loc[i, 'std_Ranking'] = pd.DataFrame(std_id_Ranking[:,1]).loc[j][0]

#        j += 1
# Нормировка min-max для Restaurant_id и Ranking 

# (x - x.min)/(x.max = x.min)



data['min_max_id'] = 0 

data['min_max_Ranking'] = 0

#data = data.sort_values(by = ['City', 'Ranking'], axis=0).reset_index(drop=True)

for City in list_of_City:

    loc_data = data[data.City == City]

    scale_features_mm = MinMaxScaler()  

    min_max_id_Ranking = scale_features_mm.fit_transform(loc_data[['Restaurant_id','Ranking']]) 

    j=0

    for i in data.index[data['City']==City]:

        data.loc[i, 'min_max_id']= pd.DataFrame(min_max_id_Ranking[:,0]).loc[j][0]

        data.loc[i, 'min_max_Ranking'] = pd.DataFrame(min_max_id_Ranking[:,1]).loc[j][0]

        j += 1
#  Посмотрим, как отображаются новые признаки, распечатаем к примеру Рим

data[data.City == 'Rome'].head(10)
data['min_max_id'].hist(bins=100)
data['min_max_Ranking'].hist(bins=100)
#data['std_id'].hist(bins=100)
#data['std_Ranking'].hist(bins=100)
# Признак 'relative_rank' - отношение показателя Ranking к количеству ресторанов в городе 'rest_number_in_City'



data['relative_rank'] = data['Ranking'] / data['restaurant_num_in_City']
# Пизнак rewiew_in_City - количество отзывов в ресторанах города



for city in city_list:

    data.loc[data.City == city, 'rewiew_in_City'] = data['Number of Reviews'][data.City == city].sum()
# Признак relative_rank_review - отношение показателя Ranking к количеству отзывов в городе rewiew_in_City



data['relative_rank_review'] = data['Ranking'] / data['rewiew_in_City']
data['Price Range'].value_counts()
data['Price Range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')
# Price Range - кодировка (замена по словарю)



Price_Range_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3}



data = data.replace({"Price Range": Price_Range_dict})
data['Price Range'] = data.apply(lambda x: 2 if pd.isna(x['Price Range']) else x['Price Range'], axis=1)
data.Reviews.value_counts()
data['Reviews_isna'] = pd.isna(data['Reviews']).astype('uint8')
data.loc[data.Reviews=='[[], []]', 'Reviews_isna'] = 1
data.Reviews.value_counts().index[1:31]
#double_Reviews = data.Reviews.value_counts().index[1:31]
data.info(verbose=True, null_counts = True)
#for review in double_Reviews:

#    display(data[data.Reviews == review])
#len(data[data.duplicated(subset = ['City', 'Cuisine Style', 'Reviews', 'URL_TA', 'ID_TA'])])
#data[data.duplicated(subset = ['City', 'Cuisine Style', 'Reviews', 'URL_TA', 'ID_TA'])]
#data.drop_duplicates(subset = ['City', 'Cuisine Style', 'Reviews', 'URL_TA', 'ID_TA'], inplace = True, keep = 'last')
#len(data)
#for review in double_Reviews:

#    display(data[data.Reviews == review])
# Признак Reviews_txt - тескты отзывов не раздленные



#data['Reviews_txt'] = data['Reviews'].str.findall(r'\[([^\[^\]]+)\]').str[0]
# Признаки Rev_txt_1 и Rev_txt_2 - тексты отзывов



#txt = data['Reviews_txt'].str.split("', ",expand=True)

#txt
#txt[txt[2].notnull()]
#data.loc[22098, 'Reviews']
# Выявилась строка, где сочетание симовлов, которое используем для разделения строки, встречается в тексте отзыва

# Заменяем комбинацию символов



#data.loc[22098, 'Reviews_txt'] = data['Reviews_txt'][22098].replace("', Captain", 'Captain')
#data.loc[22098, 'Reviews_txt']
# Включаем эту уникальную строку в разбивку

#txt = data['Reviews_txt'].str.split("', ",expand=True)

#txt
# Добавляем новые столбцы в наш датасет

#txt.columns = ['Rev_txt_1', 'Rev_txt_2']

#data = pd.concat([data, txt], axis=1)
# Признаки Rev_date_1 и Rev_date_2 - даты отзывов



data['Reviews_date'] = data['Reviews'].str.findall(r'\[([^\[^\]]+)\]').str[-1]

Rew_date = data['Reviews_date'].str.split(",",expand=True)

Rew_date.columns = ['Rev_date_1', 'Rev_date_2']

Rew_date['Rev_date_1'] = pd.to_datetime(Rew_date['Rev_date_1'])

Rew_date['Rev_date_2'] = pd.to_datetime(Rew_date['Rev_date_2'])

data = pd.concat([data, Rew_date], axis=1)
data.info(verbose=True, null_counts = True)
#data[data.Rev_txt_2.isna() & data.Rev_date_2.notnull()].Reviews
#Rev_to_split = data[data.Rev_txt_2.isna() & data.Rev_date_2.notnull()].Reviews_txt

#txt = Rev_to_split.str.split('", ',expand=True)

#txt
#txt_index = txt.index
#for i in txt_index:

#    data.loc[i, 'Rev_txt_1'] = txt.loc[i, 0]

#    data.loc[i, 'Rev_txt_2'] = txt.loc[i, 1]
#data.info(verbose=True, null_counts = True)
#data[data.Rev_txt_2.isna() & data.Rev_date_2.notnull()].Reviews
#data.loc[22958, 'Reviews']
#data.loc[22958, 'Rev_txt_2'] = data.loc[22958, 'Rev_txt_1'].split(',')[1]

#data.loc[22958, 'Rev_txt_1'] = 0
#data.loc[22958, 'Rev_txt_1']
#data.loc[28330, 'Reviews']
#data.loc[28330, 'Rev_txt_1'] = data.loc[28330, 'Reviews'].split("', '")[0]

#data.loc[28330, 'Reviews_txt'] = data.loc[28330, 'Reviews'].split('ju')[0]

#data.loc[28330, 'Rev_txt_2'] = data.loc[28330, 'Reviews_txt'].split("', '")[-1]
#data.loc[35933, 'Reviews']
#data.loc[35933, 'Reviews_txt'] = data.loc[35933, 'Reviews'].split('12')[0]

#data.loc[35933, 'Rev_txt_1'] = data.loc[35933, 'Reviews_txt'].split(', ')[0]

#data.loc[35933, 'Rev_txt_2'] = data.loc[35933, 'Reviews'].split(',')[1]
#data.info(verbose=True, null_counts = True)
#data.drop(['Reviews', 'Reviews_txt', 'Reviews_date'], axis = 1, inplace = True)

data.drop(['Reviews', 'Reviews_date'], axis = 1, inplace = True)
# Признак delta_days - делта между первой и последней датой отзывов, в днях



data['delta_days'] = (data['Rev_date_1'] - data['Rev_date_2']).dt.days
data.delta_days = data.apply(lambda x: 0 if pd.isna(x['delta_days']) else x['delta_days'], axis=1)
# Признак Review_weekday_1 и Review_weekday_2 - день недели написания отзыва



data['Review_weekday_1'] = data['Rev_date_1'].dt.weekday

data['Review_weekday_2'] = data['Rev_date_2'].dt.weekday
data.Review_weekday_1 = data.apply(lambda x: 0 if pd.isna(x['Review_weekday_1']) else x['Review_weekday_1'], axis=1)

data.Review_weekday_2 = data.apply(lambda x: 0 if pd.isna(x['Review_weekday_2']) else x['Review_weekday_2'], axis=1)
data.info(verbose=True, null_counts = True)
# Признак положительной окраски отзыва
data['Number of Reviews'].value_counts()
data['Number of Reviews'].hist(bins=100)
data['Number of Reviews'].max()
sns.boxplot(data=data['Number of Reviews'])
# рассчитаем границы выбросов

median = data['Number of Reviews'].median()

IQR = data['Number of Reviews'].quantile(0.75) - data['Number of Reviews'].quantile(0.25)

perc25 = data['Number of Reviews'].quantile(0.25)

perc75 = data['Number of Reviews'].quantile(0.75)

print('25-й перцентиль: {},'.format(perc25),

      '75-й перцентиль: {},'.format(perc75),

      "IQR: {}, ".format(IQR),

      "Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))



#data['Number of Reviews'].loc[data['Number of Reviews'] <= 287].hist(bins = 30, 

#                                  range = (0, 287), 

#                                  color = 'red',

#                                  label = 'Здравый смысл')



data['Number of Reviews'].loc[data['Number of Reviews'].between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins = 30, 

                                                                        range = (0, 287),

                                                                        color = 'blue',

                                                                        label = 'IQR')



plt.legend();
# Признак Number_of_Reviews_isNAN - признак отсутствия данных в Number_of_Reviews



data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
# Заполняем пустые значения на самое частое значение - "2" на основании наличия даты отзыва, 

# если встречается ресторан без даты отзыва, заполняем нулём 



data['Number of Reviews'] = data.apply(lambda x: (0 if x.Rev_date_1 == np.datetime64('NaT') else 2) if pd.isna(x['Number of Reviews']) else x['Number of Reviews'], axis=1)
data.info(verbose=True, null_counts = True)
data.drop(['URL_TA'], axis = 1, inplace = True)
data.ID_TA
data['ID_TA'] = data['ID_TA'].str.replace('d', '').astype(int)
data.ID_TA.hist(bins=100)
data['ID_TA'][data['City'] =='Rome'].hist(bins=20)
data.info(verbose=True, null_counts = True)
data['Rating'].value_counts(ascending=True).plot(kind='barh')
data['Ranking'][data['Rating'] == 5].hist(bins=100)
data['Ranking'][data['Rating'] < 4].hist(bins=100)
data.corr()
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.corr(),)
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    #df_output.drop(['City','Cuisine Style', 'Country', 'Rev_txt_1', 'Rev_txt_2', 'Rev_date_1', 'Rev_date_2'], axis = 1, inplace=True)

    df_output.drop(['City','Cuisine Style', 'Country', 'Rev_date_1', 'Rev_date_2'], axis = 1, inplace=True)

    #df_output.drop(['std_Ranking'], axis = 1, inplace=True)

    

    # ################### 2. NAN ############################################################## 

    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...

    #df_output['Number of Reviews'].fillna(0, inplace=True)

    # тут ваш код по обработке NAN

    # ....

    

    

    # ################### 3. Encoding ############################################################## 

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    #df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)

    # тут ваш код не Encoding фитчей

    # ....

    

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    # ....

    

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

    

    #object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    #df_output.drop(object_columns, axis = 1, inplace=True)

    

    return df_output
df_preproc = preproc_data(data)

df_preproc.sample(10)
df_preproc.info(verbose=True, null_counts = True)
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

#y_pred = np.round(y_pred*2)/2
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