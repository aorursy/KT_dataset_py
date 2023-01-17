import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler





df=pd.read_csv('../input/data-preprocessing/data.csv')

df.head()
df.describe()
scaler = MinMaxScaler()

df_values = df.values

df_scaled = scaler.fit_transform(df_values)

normalized_df = pd.DataFrame(df_scaled)

normalized_df.head()
normalized_df.describe()
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

df_values = df.values

df_std = std_scaler.fit_transform(df_values)

standardized_df = pd.DataFrame(df_std)

standardized_df.head()
standardized_df.describe()
df=pd.read_csv('../input/sample-data1/2015.csv')

df.head()
df.info()
df.head()
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

df['Country'] = lb.fit_transform(df['Country'])

df.head()

df.info()
df = pd.get_dummies(df,columns=['Country'], prefix=['Co'])

df.head()
df.info()