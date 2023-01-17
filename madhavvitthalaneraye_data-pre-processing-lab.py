import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
df =pd.read_excel("../input/multiline/companysalar.xlsx")
df.columns
df.head(10)
df.tail()
Scaler = MinMaxScaler()

df_values = df.values

df_valued = Scaler.fit_transform(df_values)

normalized_df = pd.DataFrame(df_valued)

normalized_df
df.describe()
std_scaler = StandardScaler()

df_values = df.values

df_std = std_scaler.fit_transform(df_values)

std_df = pd.DataFrame(df_std)

std_df
df.describe()
print(df.isnull().sum())
df.Salary = df.Salary.fillna("unknown")

print(df.isnull().sum())
print(df.shape)

print("\n")

print(df.dtypes)
df.info()

print('_'*40)
df = pd.read_excel("../input/product/product.xlsx")

df
df.info()
lb = LabelEncoder()

df['Country'] = lb.fit_transform(df['Country'])

df
df.info()
df.info()
new = {'No':0, 'Yes':1}

df.Purchased = df.Purchased.map(new)

df
df.info()