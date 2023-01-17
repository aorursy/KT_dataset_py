import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('../input/predictingese/amsPrediction - Sheet1.csv')

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
dt = pd.read_csv('../input/predictingese/std-cat.csv')

dt.head()
new = {'F':0,'M':1}

dt = dt.replace({'Gender':new})

dt.info()
dt.head()
from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder()

dt['State'] = lb.fit_transform(dt['State'])

dt.head()
dt = pd.get_dummies(dt, columns=['Category'],prefix = ['Cat'])

dt.head()
dt.info()