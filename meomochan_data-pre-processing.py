import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler



df = pd.read_csv('/kaggle///input/data-preprocessing/amsPred-Sheet1.csv')
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
ef = pd.read_csv('/kaggle/input/data-preprocessing/std-cat.csv') 

ef.head()
ef.info()
new = {'F':0 ,'M':1}

ef = ef.replace({'Gender':new})

ef.info()
ef.describe()
ef.head()
from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder()

ef['State'] = lb.fit_transform(ef['State'])

ef.head()
ef.info()
ef = pd.get_dummies(ef , columns=['Category'] , prefix =['cat'])
ef.head()
ef.info()