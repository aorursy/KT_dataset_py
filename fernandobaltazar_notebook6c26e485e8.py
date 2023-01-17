import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib as mp
data = pd.read_csv("../input/data-housing/Housing.csv")

data.head()
# Preferí hacer esta antes ya que no se usará esa columna

del data["Id"]
# data.shape[1]

len(data.columns)

# Hay 80 columnas (ya que se elimino Id).
data.select_dtypes("number").describe()
print("Número de columnas numéricas " , len(data.select_dtypes("number").columns))

print("\nNúmero de columnas categóricas " , len(data.select_dtypes("object").columns))
data.shape
# El 50% de las filas es 730 (data.shape[1]/2)

missing_by_column = data.isnull().sum()

missing_50 = missing_by_column.loc[missing_by_column > 730]

# print("Número de columnas con más del 50% de registros NaN: ", missing_50.size)

print("Variables con más del 50% de valores NaN \n", missing_50)
data.select_dtypes("object").isnull().sum()
data.select_dtypes("number").isnull().sum()
# En realidad YrSold no tiene valores faltantes pero se haría así.

data["YrSold"].fillna(round(data["YrSold"].mean()))
data["LotFrontage"].fillna(data["LotFrontage"].median())
from sklearn.preprocessing import OneHotEncoder
# creating instance of OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
pd.DataFrame(enc.fit_transform(data[['SaleCondition']]).toarray())
data.columns
data[['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']].head()
data["TotalBath"] = pd.Series(data.FullBath + (data.HalfBath*0.5))

data["TotalBath"].head()
data["LotArea"].describe()
sns.boxplot(y=data["LotArea"])
data["SalePrice"].describe()
sns.boxplot(y=data["SalePrice"])