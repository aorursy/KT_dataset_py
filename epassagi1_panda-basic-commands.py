import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
data = pd.read_csv('../input/train.csv')
data.shape
print(data.head()) #similar case to data.tail(), with output being the last 5 rows.
print(data.dtypes)
data.isnull().sum()
print(data.PassengerId.nunique())
print(data.describe())
print(data.Pclass.value_counts())
data.sort_values('PassengerId', ascending=False, inplace=True)
#data = data.sort_values('PassengerId', ascending=False) #other alternative, same output
print(data.head())
data.drop_duplicates('Sex', keep='first', inplace=True)
#data = data.drop_duplicates('Sex', keep='first') #also alretnatives for the absence of 'inplace=True'
print(data.shape)
print("Here are all the data left:\n")
print(data)
data = pd.read_csv('../input/train.csv')
print("Number of null entries in Age column before fill in is: "+ str(data['Age'].isnull().sum()) + "\n")
avr_age = float(data['Age'].mean())
data['Age'].fillna(avr_age, inplace=True) #avr_age arguments contains the float value for the replacement
print("Number of null entries in Age column after fill in is: "+ str(data['Age'].isnull().sum()))
data_gp = data.groupby(['Survived', 'Sex']).agg({'PassengerId': 'count',
                                      'Age': 'mean'
                                      })
data_gp.reset_index(inplace=True) #to retain the index numbering for the resulting table
print(data_gp)