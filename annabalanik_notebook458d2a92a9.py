# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sklearn
import io
import re
import math

import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
data_train = pd.read_csv('/kaggle/input/finalone/data_train_items(2).csv')
test = pd.read_csv('/kaggle/input/finalone/data_test_items(2).csv')
list_for_median = ['total_cost_min', 'total_cost_max', 'total_cost_sum', 'total_cost_mean', 'total_weight_min',
                   'total_weight_max', 'total_weight_sum', 'total_weight_mean', 'shipment_duration_time_sum', 
                'order_time_min', 'order_time_max', 'order_time_sum', 
                   'order_time_mean', 'rate_min', 'rate_max', 'rate_mean', ]


list_for_zero = ['order_duration_time_min', 'order_duration_time_max', 'order_duration_time_sum', 
                 'order_duration_time_mean', 'shipment_duration_time_min', 'shipment_duration_time_max',
                 'promo_total_min', 'promo_total_sum',
                 'is_complete_sum', 'is_canceled_sum', 'is_web_max', 'is_app_max', 'is_windows_max', 
                 'is_linux_max', "('is_push', 'max')", "('is_email', 'max')", 
                 "('is_sms', 'max')", "('is_hidden', 'max')", "('is_sale', 'max')"
                   ]
data_train[list_for_median] = data_train[list_for_median].fillna(data_train[list_for_median].median())
data_train[list_for_zero] = data_train[list_for_zero].fillna(0)
num_feat_test = [cname for cname in data_train.columns if
                    data_train[cname].dtype != "object" and cname not in list_for_median and cname not in list_for_zero]
num_feat_test
#num_feat_test.remove('target') 
num_feat_test.remove('age') 
num_feat_test.remove('gender') 
num_feat_test.remove('phone_id') 
#'phone_id', 'gender', 'age'])
num_feat_test
num_feat_test.remove('target') 
num_feat_test
data_train[num_feat_test] = data_train[num_feat_test].fillna(data_train[num_feat_test].median())
test[num_feat_test] = test[num_feat_test].fillna(0) 
test[list_for_median] = test[list_for_median].fillna(data_train[list_for_median].median())
test[list_for_zero] = test[list_for_zero].fillna(0) 
feature_drop = [ 'is_app_max',
                 'is_linux_max', "('is_push', 'max')", "('is_email', 'max')", 
                 "('is_sms', 'max')", "('is_hidden', 'max')", "('is_sale', 'max')"]
data_train = data_train.drop(feature_drop, axis=1)
test = test.drop(feature_drop, axis=1)
data_train.age.isnull().sum()
data_train.age=data_train.age.fillna(0)
test.age=test.age.fillna(0)
test.gender.unique()
test.gender=test.gender.fillna(2)
test.gender = test.gender.astype('int')
test.gender.unique()
data_train.gender=data_train.gender.fillna(2)
data_train.gender = data_train.gender.astype('int')
data_train.gender.unique()
data_train.city=data_train.city.fillna('NaN')
data_train['city'].replace(['NaN', 'Москва', 'Московская Область', 'Казань', 'Ростов-на-Дону',
       'Ульяновск', 'Краснодар', 'Екатеринбург', 'Санкт-Петербург',
       'Нижний Новгород', 'Уфа', 'Воронеж', 'Самара', 'Волгоград', 'Омск',
       'Новосибирск', 'Красноярск', 'Пермь', 'Тюмень', 'Иркутск',
       'Калининград', 'Рязань', 'Челябинск', 'Ижевск', 'Магнитогорск',
       'Оренбург', 'Томск', 'Кемерово', 'Барнаул', 'Липецк', 'Ярославль',
       'Владимир', 'Калуга', 'Астрахань', 'Курск', 'Мурманск', 'Киров',
       'Тула', 'Тольятти', 'Саратов', 'Вологда', 'Ставрополь', 'Тверь',
       'Новокузнецк', 'Нарьян-Мар', 'Сургут', 'Набережные Челны',
       'Новороссийск', 'Махачкала', 'Курган', 'Ханты-Мансийск',
       'Альметьевск', 'Пятигорск', 'Орел', 'Смоленск', 'Нижний Тагил',
       'Майкоп', 'Пенза', 'Чебоксары', 'Белгород', 'Брянск',
       'Петропавловск-Камчатский', 'Петрозаводск', 'Тамбов',
       'Архангельск', 'Сыктывкар', 'Псков', 'Кызыл', 'Абакан', 'Иваново',
       'Йошкар-Ола', 'Бийск', 'Улан-Удэ', 'Саранск', 'Южно-Сахалинск',
       'Владивосток', 'Черкесск', 'Ноябрьск', 'Тобольск', 'Владикавказ',
       'Стерлитамак', 'Кострома', 'Великий Новгород', 'Элиста', 'Нальчик',
       'Орск', 'Благовещенск', 'Горно-Алтайск', 'Магадан', 'Юрга',
       'Якутск', 'Чита', 'Хабаровск', 'Биробиджан', 'Грозный', 'Анадырь', 'NaN'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                      21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                                      41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
                                      61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,
                                      81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96],inplace=True)

data_train.city = data_train.city.astype('int')
data_train.city.unique()
test.city.isnull().sum()
test.city.unique()
test.city=test.city.fillna('NaN')
test['city'].replace(['NaN', 'Москва', 'Московская Область', 'Казань', 'Ростов-на-Дону',
       'Ульяновск', 'Краснодар', 'Екатеринбург', 'Санкт-Петербург',
       'Нижний Новгород', 'Уфа', 'Воронеж', 'Самара', 'Волгоград', 'Омск',
       'Новосибирск', 'Красноярск', 'Пермь', 'Тюмень', 'Иркутск',
       'Калининград', 'Рязань', 'Челябинск', 'Ижевск', 'Магнитогорск',
       'Оренбург', 'Томск', 'Кемерово', 'Барнаул', 'Липецк', 'Ярославль',
       'Владимир', 'Калуга', 'Астрахань', 'Курск', 'Мурманск', 'Киров',
       'Тула', 'Тольятти', 'Саратов', 'Вологда', 'Ставрополь', 'Тверь',
       'Новокузнецк', 'Нарьян-Мар', 'Сургут', 'Набережные Челны',
       'Новороссийск', 'Махачкала', 'Курган', 'Ханты-Мансийск',
       'Альметьевск', 'Пятигорск', 'Орел', 'Смоленск', 'Нижний Тагил',
       'Майкоп', 'Пенза', 'Чебоксары', 'Белгород', 'Брянск',
       'Петропавловск-Камчатский', 'Петрозаводск', 'Тамбов',
       'Архангельск', 'Сыктывкар', 'Псков', 'Кызыл', 'Абакан', 'Иваново',
       'Йошкар-Ола', 'Бийск', 'Улан-Удэ', 'Саранск', 'Южно-Сахалинск',
       'Владивосток', 'Черкесск', 'Ноябрьск', 'Тобольск', 'Владикавказ',
       'Стерлитамак', 'Кострома', 'Великий Новгород', 'Элиста', 'Нальчик',
       'Орск', 'Благовещенск', 'Горно-Алтайск', 'Магадан', 'Юрга',
       'Якутск', 'Чита', 'Хабаровск', 'Биробиджан', 'Грозный', 'Анадырь', 'Братск', 'Прокопьевск', 'NaN'],
                     [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                      21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                                      41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
                                      61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,
                                      81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98],inplace=True)
print(data_train.columns)
print(test.columns)
a = data_train.columns.tolist()
b=test.columns.tolist()

c = [i for i in a if i not in b]
d = [i for i in b if i not in a]
print(c, d)
last_drop = ['price_max_max',       
'price_mean_max',      
'discount_mean_max',    
'cancelled_min_max',    
'cancelled_min_sum',    
'price_max_mean',      
'price_min_max',        
'quantity_min_max',    
'discount_min_sum',    
'discount_min_max',    
'price_max_min','replaced_min_max',
'replaced_min_sum', 'price_mean_mean', 'price_min_mean',
]

#data_train = data_train.drop(last_drop, axis=1)
#test = test.drop(last_drop, axis=1)
ytrain = data_train.target.values
data_train=data_train.drop(['target', 'order_completed_at'], axis=1)
print(data_train.shape)
print(test.shape)
test1 = test.values
xtrain = data_train.values
from sklearn.ensemble import RandomForestClassifier 
model_rf = RandomForestClassifier(n_estimators=600, max_depth=3, random_state=42)
model_rf.fit(xtrain, ytrain)
pred = model_rf.predict(test1)
sub = pd.DataFrame({'Id':test.Id, 'Predicted': pred})
sub.Predicted = sub.Predicted.astype('int')
sub.to_csv('my_submission_new.csv', index=False, header=True, sep=",")
