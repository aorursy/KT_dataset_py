import numpy as np 
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.info()
df
#Входные данные - колонка в unixtime для датафрейма с названием df.
#Выходные данные - в df колонка из unixtime преобразуется в datetime.
def unixtime_to_datetime(unixtime_colomn):
    df[unixtime_colomn] = pd.to_datetime(df[unixtime_colomn], unit='s')
df['Datetime'] = pd.to_datetime(df['Datetime'],format='%d-%m-%Y %H:%M')

df['year']=df['Datetime'].dt.year 
df['month']=df['Datetime'].dt.month 
df['day']=df['Datetime'].dt.day

df['dayofweek_num']=df['Datetime'].dt.dayofweek  
df['dayofweek_name']=df['Datetime'].dt.weekday_name

data.head()
df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
df
def fill_value (column_na): #На вход передаётся таблица с NaN-значениями
    df[column_na] = df[column_na].fillna(-999) #пустые значения можно выделить значением, которое не встречается в df, таким образом мы указываем, что в данном значении раньше был null
    
#Пример вставки
fill_value('Age')
df
def fill_mean (column_na, groupby_parameter): #На вход подаётся column_na - столбец с пропусками, groupby_parameter - параметр по которому осуществляется группирока, для нахождения среднего
    df[column_na] = df.groupby(groupby_parameter)[column_na].transform(
        lambda grp: grp.fillna(np.mean(grp))
    )
# Пример использования
fill_mean('Age', 'Sex')
df
def fill_similar(column_na):#На вход передаётся таблица с NaN-значениями
    df[column_na] = df[column_na].ffill().bfill()
# Пример использования
fill_similar('Cabin')
df
sns.boxplot(x=df['Age'])
df['Age'].describe()
def find_outburst(column):
    low_fence = df[column].quantile(0.10)
    high_fence = df[column].quantile(0.90)
    df_out = df.loc[(df[column] > low_fence) & (df[column] < high_fence)]
    return df_out
df_out = find_outburst('Age')
sns.boxplot(x=df_out['Age'])
def new_feat_outburst(column):
    low_fence = df[column].quantile(0.10)
    high_fence = df[column].quantile(0.90)
    df['Outburst'] = np.where((df[column] > low_fence) & (df[column] < high_fence), 0, 1)
    return df
new_df = new_feat_outburst('Age')
new_df
df
def find_cat(df):
    for name in df.columns:
        cat_feat = ''
        cat_feat += name
        if (type(df[name][0])== str): #Проверяем, что в элементе 0 у столбца строка
            cat_feat += ' строка'
        if (df[name].nunique()<=10): #Находим количество уникальных значений в датафрейме, число 10 изменяем при необходимости - это поможет выбрать категориальные значения
            cat_feat += ' менее 10 уникальных значений'
        if (cat_feat!=name):
            print(cat_feat)
            
find_cat(df)
# Объединение 2ух значений (конъюнция)
def make_feat_conj(df, cat1, cat2):
    df[cat1 + '+' + cat2] = df[cat1].astype(str) + '+' + df[cat2].astype(str)
    return df
make_feat_conj(df, 'Sex', 'Embarked')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df.Sex)
df['Sex_le'] = le.transform(df.Sex)
df
df
#Работает только с целочисленными значениями
from sklearn.preprocessing import OneHotEncoder
just_dummies = pd.get_dummies(df['Embarked'])
#где df['dummy'] - столбец, который нужно закодировать
#Конкатенируешь к датафрейму df закодированные столбцы
step_1 = pd.concat([df, just_dummies], axis=1)      
#Удаляешь столбец на основе которого строил признаки
step_1.drop(['Embarked'], inplace=True, axis=1)
df
def code_mean(df, cat_feat, real_feat):
    return(df[cat_feat].map(df.groupby(cat_feat)[real_feat].mean()))

df['Sex_Age_mean'] = code_mean(df, 'Sex', 'Age')
df
