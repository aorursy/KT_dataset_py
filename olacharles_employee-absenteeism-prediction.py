import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
raw_data = pd.read_csv('/kaggle/input/absenteeism-data/Absentdata.csv')
raw_data
df = raw_data.copy()
pd.options.display.max_columns = None

pd.options.display.max_rows = None
df.info()
df=df.drop(['ID'], axis=1)
df_no_age = df.drop(['Age'], axis=1)
sorted(df['Reason for Absence'].unique())
df.info()
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
reason_columns.head()
reason_columns['Check'] = reason_columns.sum(axis=1)
#reason_columns
reason_columns['Check'].unique()
reason_columns=reason_columns.drop(['Check'], axis=1)
age_dummies = pd.get_dummies(df.Age)
age_dummies['check'] = age_dummies.sum(axis=1)
age_dummies.head()
df.columns.values
df=df.drop(['Reason for Absence'], axis=1)
reason_columns.columns.values
reason_type_1 = reason_columns.iloc[:,0:14].max(axis=1)

reason_type_2 = reason_columns.iloc[:,14:17].max(axis=1)

reason_type_3 = reason_columns.iloc[:,17:21].max(axis=1)

reason_type_4 = reason_columns.iloc[:,21:].max(axis=1)
df = pd.concat([df, reason_type_1,reason_type_2,reason_type_3,reason_type_1], axis=1)
df.head()
df.columns.values
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_type_1', 'Reason_type_2','Reason_type_3', 'Reason_type_4']
df.columns=column_names
df.head()
age_dummies = age_dummies.drop(['check'], axis=1)
df_concatenate = pd.concat([df_no_age, age_dummies], axis=1)
df_concatenate.head()
column_names_reordered = ['Reason_type_1', 'Reason_type_2','Reason_type_3', 'Reason_type_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours']
df= df[column_names_reordered]

df.head()
df_concatenate.columns.values
col = ['Reason for Absence', 'Date', 'Transportation Expense',

       'Distance to Work', 'Daily Work Load Average', 'Body Mass Index',

       'Education', 'Children', 'Pets',27,

       28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 43, 46, 47, 48,

       49, 50, 58 , 'Absenteeism Time in Hours']
df_concatenate = df_concatenate[col]
df_concatenate.head()
df.Education.value_counts()
df_reason_mod = df.copy()
df_reason_mod.head()
df_checkpoint = df_concatenate.copy()
df_reason_mod.Education.value_counts()
type(df_reason_mod.Date[0])
df_reason_mod['Date'] = pd.to_datetime(df_reason_mod.Date, format= '%d/%m/%Y')
type(df_reason_mod.Date[0])
df_reason_mod.Education.value_counts()
list_months = []

list_months
for i in range(df_reason_mod.shape[0]):

    list_months.append(df_reason_mod['Date'][i].month)
df_reason_mod['Months'] = list_months
df_reason_mod.head(20)
df_reason_mod.Education.value_counts()
list_dayofweek = []

list_dayofweek
for i in range(df_reason_mod.shape[0]):

    list_dayofweek.append(df_reason_mod['Date'][i].weekday())
df_reason_mod['Weekday']=list_dayofweek
df_reason_mod.head(20)
df_reason_mod.Education.value_counts()
df_reason_mod= df_reason_mod.drop(['Date'], axis=1)
df_reason_mod.columns.values
newcol = ['Reason_type_1', 'Reason_type_2', 'Reason_type_3', 'Reason_type_4', 'Months',

       'Weekday','Transportation Expense', 'Distance to Work', 'Age',

       'Daily Work Load Average', 'Body Mass Index', 'Education',

       'Children', 'Pets', 'Absenteeism Time in Hours']
df_reason_mod=df_reason_mod[newcol]
df_reason_mod.head()
df_reason_mod.Education.value_counts()
df_reason_date_mod = df_reason_mod.copy()
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1,3:1,4:1})
df_reason_date_mod['Education'].value_counts()
df_preprocessed = df_reason_date_mod.copy()
df_preprocessed.head()
df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)