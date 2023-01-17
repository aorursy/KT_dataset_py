import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../input/asmpredictioncsv/amsPredictionSheet1-201009-150447 (2).csv')
df.head()
scaler = MinMaxScaler()
df_values = df.values
df_scaled = scaler.fit_transform(df_values)
normalized_df = pd.DataFrame(df_scaled)
normalized_df.head()
normalized_df.describe()
from sklearn.preprocessing import  StandardScaler
std_scaler = StandardScaler()
df_values = df.values
df_std = std_scaler.fit_transform(df_values)
standardized_df = pd.DataFrame(df_std)
standardized_df.head()

standardized_df.describe()
df.info()
new = {'F':0,'M':1}
df=df.replace({'Gender':new})
df.info()
df.head()
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df['MSE'] = lb.fit_transform(df['MSE'])
df.head()
df.info()
df = pd.get_dummies(df, columns=['HRS'],prefix=['ESE'])
df.head()
df.info()
