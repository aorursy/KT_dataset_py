import pandas as pd
import numpy as np
df =pd.read_csv("../input/train.csv")
df.head()
df.describe()
df.info()
df.dtypes
df.shape
df.isnull().sum()
df[df.Embarked.isnull()]
df['Embarked'].fillna(value='Not Known',inplace=True)
df.info()
df.isnull().sum()
df.dropna(how='any').shape
df.shape
891-733
df.dropna(subset=['Age','Cabin'],how='all').shape
df.describe()
df
%matplotlib inline
df.Fare.plot(kind='hist')
df.Fare.plot(kind='line')

df.Fare.plot(kind='box')
df.head(15)
df.Fare.describe()
# I was trying to implement that Income Class category but I am stuck at some point
