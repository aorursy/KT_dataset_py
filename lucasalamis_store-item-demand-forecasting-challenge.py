import pandas as pd
import numpy as np
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df.head()
df.info()
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df_test['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df.info()
df_test.head()
x = df[['year','month','day','store','item']]
y = df['sales']
x_test = df_test[['year','month','day','store','item']]
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x,y)
pred = lm.predict(x_test)
pred
