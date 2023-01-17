# Анализ набора телематических данных и их классификация для оценки опасного вождения водителя.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
# Набор данных для обучения.

df_data0=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')

# Набор меток для обучения.

df_Labels=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/labels/labels-part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')
# Проверим, есть ли какие-либо нулевые значения в наборе данных, если да, то удалим эти строки данных, 
# так как они могут повлиять на производительность обучения модели.

if (df_data0.isnull().sum().sum()!=0):
    df_data0.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
# Проверим, есть ли в наборе данных меток все уникальные идентификаторы бронирования.
# Если есть дублированные, то отбросим эти строки, так как они будут ломать обучение.

if ((len(df_Labels))!=(len(df_Labels['bookingID'].unique()))):
    duplicateRows_Labels=df_Labels[df_Labels.duplicated(['bookingID'],keep=False)]
    for x in range(len(duplicateRows_Labels)):
        df_Labels.drop(duplicateRows_Labels.index[x],inplace=True)
# Посмотрим, остались ли дубликаты?

duplicateRows_Labels=df_Labels[df_Labels.duplicated(['bookingID'],keep=False)]
duplicateRows_Labels
# Объединяем набор данных labels с набором данных features, используя bookingID. 
# bookingID - ключ для объединения обоих наборов данных.

df_Combined_Dataset=pd.merge(df_data0, df_Labels, on='bookingID',
         left_index=True, right_index=False, sort=False)
# Заголовки столбцов в объединенном наборе данных.

df_Combined_Dataset.columns
# Посмотрим на скомбинированный набор данных.

df_Combined_Dataset.head()
# Мы разбили объединенный набор данных на составляющие его независимые и зависимые переменные. 
# BookingID считается не релевантным в анализе и не включен в список независимых переменных.

iv=df_Combined_Dataset[['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']]
dv=df_Combined_Dataset[['label']]
# Выполним масштабирование объектов, чтобы нормализовать все переменные в сопоставимых масштабах, 
# чтобы анализ не был искажен определенными переменными, принимающими большие значения.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
iv[['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']] = sc.fit_transform(iv[['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']])
# Будем использовать метод логистической регрессии.
# Так же применим метод рекурсивного устранения признаков (RFE) 
# для автоматического выбора признаков для удаления несущественных признаков.

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

log_reg=LogisticRegression(random_state=1)
#log_reg.fit(iv_train,dv_train)
#log_reg.predict(iv_test)

# Выберем атрибуты, создав модель RFE.

rfe = RFE(log_reg, 7)
rfe = rfe.fit(iv, dv)

print(rfe.support_)
print(rfe.ranking_)
idx=iv.columns
idx
reduced_features=[]
reduced_features_withKey=['bookingID']

for i in range(len(rfe.ranking_)):
    if (rfe.ranking_[i]==1):
        reduced_features.append(idx[i])
        reduced_features_withKey.append(idx[i])
reduced_features
reduced_features_withKey
# Подгрузим полный набор данных.

df_data0=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)
df_data1=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)
df_data2=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)
df_data3=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)
df_data4=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)
df_data5=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)
df_data6=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)
df_data7=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)
df_data8=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)
df_data9=pd.read_csv('../input/car-data-management-data-set/car-data-management-data-set/features/features-part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv',usecols=reduced_features_withKey)

df_ReducedFeatures_Dataset=pd.concat([df_data0,df_data1,df_data2,df_data3,df_data4,df_data5,df_data6,df_data7,df_data8,df_data9],axis=0)

# Освободим память.

del [[df_data0, df_data1, df_data2, df_data3, df_data4, df_data5, df_data6, df_data7, df_data8, df_data9]]
gc.collect()
df_data0=pd.DataFrame()
df_data1=pd.DataFrame()
df_data2=pd.DataFrame()
df_data3=pd.DataFrame()
df_data4=pd.DataFrame()
df_data5=pd.DataFrame()
df_data6=pd.DataFrame()
df_data7=pd.DataFrame()
df_data8=pd.DataFrame()
df_data9=pd.DataFrame()
# Проверим, есть ли какие-либо нулевые значения в наборе данных.
# Если это так, то отбросим эти строки данных, поскольку они могут повлиять на обучение модели.

if (df_ReducedFeatures_Dataset.isnull().sum().sum()!=0):
    df_ReducedFeatures_Dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
# Посмотрим на полный набор данных.

df_ReducedFeatures_Dataset.head()
len(df_ReducedFeatures_Dataset)
# Объединяем набор данных labels с df_ReducedFeatures_Dataset, 
# используя bookingID в качестве ключа для объединения обоих наборов данных.

df_Combined_Dataset=pd.merge(df_ReducedFeatures_Dataset, df_Labels, on='bookingID', left_index=True, right_index=False, sort=False)

# Освободим память.

del [[df_Labels, df_ReducedFeatures_Dataset]]
gc.collect()
df_Labels=pd.DataFrame()
df_ReducedFeatures_Dataset=pd.DataFrame()
# Посмотрим заголовки столбцов в объединенном наборе данных.

df_Combined_Dataset.columns
df_Combined_Dataset.head()
# Мы разбили объединенный набор данных на составляющие его независимые и зависимые переменные. 
# BookingID считается не релевантным в анализе и не включен в список независимых переменных.

iv=df_Combined_Dataset[reduced_features]
dv=df_Combined_Dataset[['label']]
# До.

iv.head()
# Выполним масштабирование объектов, чтобы нормализовать все переменные в сопоставимых масштабах, 
# чтобы анализ не был искажен определенными переменными, принимающими большие значения.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
iv[reduced_features] = sc.fit_transform(iv[reduced_features])
# После.

iv.head()
from sklearn.model_selection import train_test_split

df1=df_Combined_Dataset['bookingID']
iv_withBookingID=pd.concat([df1,iv], axis=1)

# Разделим набор данных в соотношении 80/20 для тестов и тренировки.

iv_train_withBookingID,iv_test_withBookingID,dv_train,dv_test=train_test_split(iv_withBookingID,dv,test_size=0.2,random_state=0)

# Освободим память.

del [[df1]]
gc.collect()
df1=pd.DataFrame()
# Посмотрим тренировочные данные.

iv_train_withBookingID.head()
# Посмотрим тестовые данные.

iv_test_withBookingID.head()
# Извлечем bookingID из тестовых и тренировочных данных.

iv_train=iv_train_withBookingID[reduced_features]
iv_test=iv_test_withBookingID[reduced_features]
# Еще раз посмотрим тренировочные данные, но уже без bookingID.

iv_train.head()
# Еще раз посмотрим тестовые данные, но уже без bookingID.

iv_test.head()
# Применим алгоритм к нашим предобработанным данным.

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(random_state=1)
log_reg.fit(iv_train,dv_train)
log_reg.predict(iv_test)

from sklearn.externals import joblib
joblib.dump(log_reg, 'model.joblib')
# Запишем наши данные.

Train_results=pd.DataFrame()

Train_results['bookingID']=pd.Series(range(len(iv_test_withBookingID)))
Train_results['bookingID']=iv_test_withBookingID['bookingID'].reset_index(drop=True)
Train_results['actualRating']=dv_test['label'].reset_index(drop=True)
Train_results['predictedRating']=pd.DataFrame(log_reg.predict(iv_test)).reset_index(drop=True)
# Выведем результат.

Train_results.head()
# Оценим полученные результаты.

from sklearn.metrics import classification_report
print(classification_report(dv_test,log_reg.predict(iv_test)))
# Освободим память.

del [[iv_test,Train_results,df_Combined_Dataset, iv, dv, iv_withBookingID, iv_train_withBookingID,iv_test_withBookingID,dv_train,dv_test]]

gc.collect()
iv_test=pd.DataFrame()
Train_results=pd.DataFrame()
df_Combined_Dataset=pd.DataFrame()
iv=pd.DataFrame()
dv=pd.DataFrame()
iv_withBookingID=pd.DataFrame()
iv_train_withBookingID=pd.DataFrame()
iv_test_withBookingID=pd.DataFrame()
dv_train=pd.DataFrame()
dv_test=pd.DataFrame()