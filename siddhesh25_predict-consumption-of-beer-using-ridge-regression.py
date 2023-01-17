# import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# read the dataset

df = pd.read_csv('../input/Consumo_cerveja.csv')
df.shape
df.isna().sum() # find the number of missing values in the dataset
""" Since there are a lot of missing values as compared to the number of observations we drop the missing rows """

df = df.dropna()
df.shape
df.head()
# We need to clean the dataset by replacing the ',' by '.' and convert it to float

df['Temperatura Media (C)'] = df['Temperatura Media (C)'].str.replace(',', '.').astype(float)

df['Temperatura Maxima (C)'] = df['Temperatura Maxima (C)'].str.replace(',', '.').astype(float)

df['Temperatura Minima (C)'] = df['Temperatura Minima (C)'].str.replace(',', '.').astype(float)

df['Precipitacao (mm)'] = df['Precipitacao (mm)'].str.replace(',', '.').astype(float)
df.head()
# Drop the data column as it is irrelevent to the dataset

df.drop(['Data'], 1, inplace = True)
# save the cleaned data

# df.to_csv('cleaned_beer.csv')
df.head()
import statsmodels.api as sm
X = df.iloc[:, :-1].astype(float)

y = df['Consumo de cerveja (litros)']
model = sm.OLS(y, X).fit()

model.summary()
# final dataset for applying regression

df_final = df.iloc[:, [2, 3, 4, 5]]
X = df_final.iloc[:, :-1]

y = df_final.iloc[:, -1]
from sklearn.model_selection import train_test_split
# splitting into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.linear_model import Ridge
rr = Ridge(alpha=1)
# fitting ridge regression to the dataset

rr.fit(X_train, y_train)
# Predicting the values of the consumption of beer and saving it in y_pred

y_pred = rr.predict(X_test)
from sklearn.metrics import r2_score
# R-squared value for the model

r2_score(y_test, y_pred)
# Root Mean squared error value for the model

rmse = np.sqrt((((y_pred) - (y_test))**2).mean())

rmse