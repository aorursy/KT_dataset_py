import pandas as pd
import numpy as np
import sidetable
import plotly.express as px
import matplotlib.pyplot as plt
df = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
df.head()
df.dtypes
df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
plt.figure(figsize=(16,8))
df['floor'].plot()
numeric_columns = [x for x in df.select_dtypes(exclude='object').columns]
numeric_columns
for col in numeric_columns:
  col_name = col + '_zscore'
  df[col_name] = (df[col] - df[col].mean()) / df[col].std()
df.head()
df.shape
import seaborn as sns
df = df.drop(numeric_columns, axis='columns')
df.head()
for col in df.select_dtypes(exclude='object').columns:
  df[col] = df[(df[col] > -3) & (df[col] < 3)][col]
df.shape
df.stb.missing()
df = df.dropna()
df.select_dtypes('float64').plot(kind='box',subplots=True, layout=(5,2), sharex=True, sharey=True, figsize=(32,16))
plt.show()
df.shape
df.head()
for col in df.select_dtypes(include='object').columns:
  print(df[col].name, df[col].unique())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
  df[col] = le.fit_transform(df[col])
df.head()
from sklearn.model_selection import train_test_split
x = df.drop('total (R$)_zscore', axis='columns')
y = df['total (R$)_zscore']
x_train, x_test, y_train, y_test = train_test_split(x ,y, random_state=42, test_size=0.2)
x_train.shape
x_test.shape
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(x_train, y_train)
rf_model.score(x_test, y_test)
pred_rf = rf_model.predict(x_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(pred_rf, y_test)
mae = mean_absolute_error(pred_rf, y_test)
rmse = np.sqrt(mse)
mse, mae , rmse
