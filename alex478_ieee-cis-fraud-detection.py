import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
print('+++')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
train_identity.shape # (144233, 41)
train_identity.columns
train_identity = train_identity.fillna(0) # заменить все NaN на 0
#train_identity.isnull().sum()  # показать кол-во НАН в таблице / теперь НАН нет
#print(train_identity.iloc[:,3].nunique())
train_identity.iloc[:,:10].hist() # определение колонок где мало данных/только одно значение
# удаление колонок с малым количеством данных
print('кол-во колонок до удаления -', len(train_identity.columns), '\n')
#train_identity.info(null_counts = True)
train_columns_index = [3, 4, 6, 7, 8, 9,10, 11, 21, 22, 23, 24, 25, 26, 27, 33,] 
#список к удалению / предположительно там мало данных
train_identity_new = train_identity.drop(train_identity.columns[train_columns_index], axis='columns')

print('\n кол-во колонок после удаления -', len(train_identity_new.columns), '\n', train_identity_new.columns)

## КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ИЗМЕНЯЕМ/ object --> category   /  если заменить то не сможем использовать 
## метод get_dummies


#def categor_change(train_identity_new):  # заменяем категориальные колонки / меняем тип данных
#    for i in train_identity_new.columns: # выбрать все столбцы где тип данных объект
#        if train_identity_new[i].dtypes == 'object': # выбираем колонки с типом object
#            train_identity_new[i] = train_identity_new[i].astype('category') # заменим тип на категориальные
#categor_change(train_identity_new)
#train_identity_new.info()
#train_identity_new['DeviceInfo'].count()
#train_identity_new.groupby('DeviceInfo').size() / len(df) # надо ли удалить этот признак
# значения "0" - 17% , 83% - остальных
# категор колонки - удалим ненужные и изменим нужные
# 1. удалим категор колонки
list_kategor = ['id_12','id_15','id_16','id_28','id_29','id_30','id_31','id_34','id_35','id_36','id_37','id_38','DeviceInfo', 'DeviceType']
train_identity_new = train_identity_new.drop( list_kategor, axis = 1)


#train_identity_new.info()

# 2. изменим на бинарн
#train_identity_new = train_identity_new.drop( 'DeviceInfo', axis =1) # удалим , т.к. там 1787 значений str
# ф.(get_dummies) изменяет тип "объект" а не "категор." поэтому не обязательно переводить в тип категор.
#train_identity_new = pd.get_dummies(train_identity_new, columns= ['DeviceType']) # результат стал хуже
#train_identity_new.head()
#train_identity_new.info()
print('\n кол-во колонок после удаления -', len(train_identity_new.columns), '\n', train_identity_new.columns)
print(train_identity_new.shape) # (144233, 11)
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_transaction.shape  # (590540, 394)
#train_identity.shape # (144233, 41)
train_transaction_new = train_transaction.fillna(0) # заменить все NaN на 0
#train_transaction_new.isnull().sum()  # показать кол-во НАН в таблице / теперь НАН нет
train_transaction_new.columns
train_transaction_new.info()
#train_transaction_new.columns.values
for i in train_transaction.columns: # узнаем какие колонки с типом данных  - объект
    if train_transaction[i].dtypes == 'object':
        print(i, 'object')
train_transaction_new['vv'] = train_transaction_new.filter(like = 'V').sum(1)/len(train_transaction_new)
# сложение значений колонок / в долях
train_transaction_new['vv']
#train_transaction_new['vv'].hist()
#train_transaction_new['vv'].nunique()

# удаляем или нет 
list_drop_V = train_transaction_new.filter(like = 'V').columns
list_drop_V
train_transaction_new = train_transaction_new.drop(list_drop_V, axis = 1)
train_transaction_new.columns
train_transaction_new['cc'] = train_transaction_new.filter(like = 'C').sum(1)/len(train_transaction_new)
# сложение значений колонок / в долях
#train_transaction_new['cc'].hist()
#train_transaction_new['cc'].nunique()

# удаляем / или нет 
list_drop_C = train_transaction_new.filter(like = 'C').columns
list_drop_C
train_transaction_new = train_transaction_new.drop(list_drop_C, axis = 1)
train_transaction_new.columns
train_transaction_new = train_transaction_new.rename(columns= {'TransactionID':'Transaction_id'})
train_transaction_new['dd'] = train_transaction_new[1:].filter(like = 'D').sum(1)/len(train_transaction_new)
# сложение значений колонок / в долях
train_transaction_new['dd']
#train_transaction_new['dd'].hist()
#train_transaction_new['dd'].nunique()

# удаляем или нет 
list_drop_D = train_transaction_new.filter(like = 'D').columns
list_drop_D
train_transaction_new = train_transaction_new.drop(list_drop_D, axis = 1)
train_transaction_new = train_transaction_new.rename(columns= {'Transaction_id':'TransactionID'})
train_transaction_new.columns
train_transaction_new = train_transaction_new.drop(['P_emaildomain', 'R_emaildomain'], axis = 1)
train_transaction_new.head(3)
#train_transaction_new.filter(like = 'M').info() # категор.колонки
train_transaction_new.filter(like = 'M').head()

train_transaction_new = pd.get_dummies(train_transaction_new) # заменим "М" 
print(train_transaction_new.columns)
train_transaction_new.head(3)
# тут другие категор признаки
## КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ИЗМЕНЯЕМ/ object --> category
def categor_change(train_transaction_new):  # заменяем категориальные колонки / меняем тип данных
    for i in train_transaction_new.columns: # выбрать все столбцы где тип данных объект
        if train_transaction_new[i].dtypes == 'object': # выбираем колонки с типом object
            train_transaction_new[i] = train_transaction_new[i].astype('category') # заменим тип на категориальные

categor_change(train_transaction_new)

train_transaction_new.info()
# КАТЕГОР/признаки  - удалим или заменим
list_kategor = train_transaction_new.select_dtypes(include= ['category']).columns # список категор.столбцов
list_kategor

# 1. удаляем
train_transaction_new = train_transaction_new.drop(list_kategor, axis = 1)
train_transaction_new.info()

# 2. заменим 
#df = pd.get_dummies(train_transaction_new, columns= list_kategor)
#df.head()
#df.info()
print(train_transaction_new.shape) # (590540, 14)
print(train_identity_new.shape) # (144233, 14)

#print(train_transaction_new.columns)
#print(train_transaction.columns)
#print(train_identity_new.columns)
#print(train_identity.columns)
train = train_transaction_new.merge(train_identity_new, how = 'left')
train.columns
train.shape # (590540, 62) 
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv') # преобразуем в данные в таблицу
test_identity.shape # (141907, 41)
test_identity.columns
test_identity = test_identity.fillna(0) # заменить все NaN на 0
#train_identity.isnull().sum()  # показать кол-во НАН в таблице / теперь НАН нет
# удаление колонок с малым количеством данных
print('кол-во колонок до удаления -', len(test_identity.columns), '\n')
#test_identity.info(null_counts = True)
test_columns_index = [3, 4, 6, 7, 8, 9,10, 11, 21, 22, 23, 24, 25, 26, 27, 33,] 
#список к удалению / предположительно там мало данных
test_identity_new = test_identity.drop(test_identity.columns[test_columns_index], axis='columns')

print('\n кол-во колонок после удаления -', len(test_identity_new.columns), '\n', test_identity_new.columns)
## КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ИЗМЕНЯЕМ/ object --> category
#def categor_change(test_identity_new):  # заменяем категориальные колонки / меняем тип данных
#    for i in test_identity_new.columns: # выбрать все столбцы где тип данных объект
#        if test_identity_new[i].dtypes == 'object': # выбираем колонки с типом object
#            test_identity_new[i] = test_identity_new[i].astype('category') # заменим тип на категориальные
#categor_change(test_identity_new)
#test_identity_new.info()
# категор колонки изменим - 
# 1. удалим категор колонки
list_kategor = ['id-12','id-15','id-16','id-28','id-29','id-30','id-31','id-34','id-35','id-36','id-37','id-38','DeviceInfo', 'DeviceType']
test_identity_new = test_identity_new.drop( list_kategor, axis = 1)
#test_identity_new.info()


# 2. изменим на бинарн 
# ф. изменяет тип "объект" не "категор.""
#test_identity_new = pd.get_dummies(test_identity_new, columns= ['DeviceType'])
#test_identity_new = pd.get_dummies(test_identity_new, columns= list_kategor)
#test_identity_new.head()
#test_identity_new.info()
print('\n кол-во колонок после удаления -', len(test_identity_new.columns), '\n', test_identity_new.columns)
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_transaction.shape  # (506691, 393)
#test_identity.shape # (144233, 41)
test_transaction_new = test_transaction.fillna(0) # заменить все NaN на 0
#test_transaction_new.isnull().sum()  # показать кол-во НАН в таблице / теперь НАН нет
test_transaction_new.columns
test_transaction_new.info()
#test_transaction_new.columns.values
#for i in test_transaction.columns: # узнаем какие колонки с типом данных  - объект
#    if test_transaction[i].dtypes == 'object':
#        print(i, 'object')
test_transaction_new['vv'] = test_transaction_new.filter(like = 'V').sum(1)/len(test_transaction_new)
# сложение значений колонок / в долях
#test_transaction_new['vv'].hist()
#test_transaction_new['vv'].nunique()

# удаляем или нет 
list_drop_V = test_transaction_new.filter(like = 'V').columns
list_drop_V
test_transaction_new = test_transaction_new.drop(list_drop_V, axis = 1)
test_transaction_new.columns
test_transaction_new['cc'] = test_transaction_new.filter(like = 'C').sum(1)/len(train_transaction_new)
# сложение значений колонок / в долях
#test_transaction_new['cc'].hist()
#test_transaction_new['cc'].nunique()

# удаляем / или нет 
list_drop_C = test_transaction_new.filter(like = 'C').columns
list_drop_C
test_transaction_new = test_transaction_new.drop(list_drop_C, axis = 1)
test_transaction_new.columns
test_transaction_new = test_transaction_new.rename(columns= {'TransactionID':'Transaction_id'})
test_transaction_new['dd'] = test_transaction_new[1:].filter(like = 'D').sum(1)/len(test_transaction_new)
# сложение значений колонок / в долях

#test_transaction_new['dd'].hist()
#test_transaction_new['dd'].nunique()

# удаляем или нет 
list_drop_D = test_transaction_new.filter(like = 'D').columns
list_drop_D
test_transaction_new = test_transaction_new.drop(list_drop_D, axis = 1)
test_transaction_new = test_transaction_new.rename(columns= {'Transaction_id':'TransactionID'})
test_transaction_new.columns
test_transaction_new = test_transaction_new.drop(['P_emaildomain', 'R_emaildomain'], axis=1)
test_transaction_new.head(3)
#train_transaction_new.filter(like = 'M').info() # категор.колонки
test_transaction_new.filter(like = 'M').head()
test_transaction_new = pd.get_dummies(test_transaction_new)
print(test_transaction_new.columns)
test_transaction_new.head(3)
# тут другие категор признаки
## КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ ИЗМЕНЯЕМ/ object --> category
def categor_change(test_transaction_new):  # заменяем категориальные колонки / меняем тип данных
    for i in test_transaction_new.columns: # выбрать все столбцы где тип данных объект
        if test_transaction_new[i].dtypes == 'object': # выбираем колонки с типом object
            test_transaction_new[i] = test_transaction_new[i].astype('category') # заменим тип на категориальные

categor_change(test_transaction_new)

test_transaction_new.info()
# КАТЕГОР/признаки  - удалим или заменим
list_kategor = test_transaction_new.select_dtypes(include= ['category']).columns # список категор.столбцов
list_kategor

# 1. удаляем
test_transaction_new = test_transaction_new.drop(list_kategor, axis = 1)
test_transaction_new.info()

# 2. заменим 
#df = pd.get_dummies(test_transaction_new, columns= list_kategor)
#df.head()
#test_transaction_new.info()
print(test_transaction_new.shape) # (506691, 13)
print(test_identity_new.shape) # (141907, 14) - удалил категор колонки

print(test_transaction_new.columns)
print(test_identity_new.columns)
test = test_transaction_new.merge(test_identity_new, how = 'left')
print(test.shape) # 
test.columns

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import preprocessing 
train.columns

train = train.drop(['card6_debit or credit'], axis = 1) # удалить
#train
X = train.drop('isFraud', axis = 1)
y = train['isFraud']
X = X.fillna(0)
print(train.shape) # (590540, 62)
print(test.shape) #(506691, 60)

from sklearn.metrics import roc_auc_score # выбор метрики
def metric(y_pred,y_true): # y_pred - истинные  метки, y_true - целевые показатели
    return roc_auc_score(y_pred,y_true) 
from sklearn.model_selection import train_test_split

X_scale = preprocessing.scale(X) # масштабирование / нормализация / убрать выбросы и пр
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=46, random_state=42)
print('clf')
clf.fit(X_train, y_train)
#w = cross_val_score(clf, X_scale, y, cv=5 )#kf, scoring='roc_auc')
#w.mean()
print('clf.fit ')
y_pred = clf.predict(X_test)
print('clf.predict ')
#metric(y_pred, y_test)
print('roc_auc - ', roc_auc_score(y_pred,y_test))
# roc_auc -  0.9770113294089217
#test.shape # (506691, 60)
test = test.fillna(0)
#test.info()
y_pred_test = clf.predict(test)
y_pred_test
len(y_pred_test)
# cross_val_score
#cross_val_score(clf, X_train, y_train, cv= 5) # array([0.95879192, 0.95662348, 0.95970013, 0.95935347, 0.95835681])
# результат стал хуже

#for i in range(1, 51): # поиск лучшего значения n_estimators 
#    clf = RandomForestClassifier(n_estimators=i, random_state=42)
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_test)
#    metric(y_pred, y_test)
#    print(i, ' - ', roc_auc_score(y_pred,y_test))
    
# 42 - 0.9636575450281896    
# 46 - 0.969602199093023
# Проверка
#for i in range(1, 51): # использование крос валидации / проверка
#    clf = RandomForestClassifier(n_estimators=i, random_state=42)
#    w = cross_val_score(clf, X_scale, y, cv=5 , scoring='roc_auc') # разделение на 5 частей
#    print(i, ' - ', w.max())
# от 0,5378244.... до    0.7185081454990827 

submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
#submission.shape # (506691, 2)
submission.head(3)
sample_submission['isFraud'] = y_pred_test
sample_submission.to_csv('submit.csv', index= False)
sample_submission.head(20)
#print(sample_submission.shape # (506691, 2)