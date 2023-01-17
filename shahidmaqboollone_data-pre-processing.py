import pandas as pd
import numpy as np

df = pd.read_csv('../input/students-data-for-mlr/amsPredictionSheet1-201009-150447.csv')
df
df.describe()
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
df_values = df.values
df_valued = Scaler.fit_transform(df_values)
normalized_df = pd.DataFrame(df_valued)
normalized_df
normalized_df.describe()
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
df_values = df.values
df_std = std_scaler.fit_transform(df_values)
std_df = pd.DataFrame(df_std)
std_df
std_df.describe()
df1 = pd.read_csv('../input/categorical-data/stdcat-201010-101522.csv')
df1
df1.info()
new = {'F':0, 'M':1}
df1.Gender = df1.Gender.map(new)
df1
df1.info()
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df1['State'] = lb.fit_transform(df1['State'])
df1
df1.info()
df1 = pd.get_dummies(df1, columns = ['Category'], prefix = ['Cat'])
df1
df1.info()