import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/MiningProcess_Flotation_Plant_Database.csv",decimal=",",parse_dates=["date"],infer_datetime_format=True).drop_duplicates()
df.head()
df.shape
df = df.dropna()
df.shape
df.describe()
plt.figure(figsize=(30, 25))
p = sns.heatmap(df.corr(), annot=True)
df = df.drop(['date', '% Iron Concentrate', 'Ore Pulp pH', 'Flotation Column 01 Air Flow', 'Flotation Column 02 Air Flow', 'Flotation Column 03 Air Flow'], axis=1)
df.head()
Y = df['% Silica Concentrate']
X = df.drop(['% Silica Concentrate'], axis=1)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = pd.DataFrame(min_max_scaler.fit_transform(X), columns=X.columns)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
_ = reg.fit(X_train, Y_train)
predictions = reg.predict(X_test)
predictions
from sklearn.metrics import mean_squared_error
error = mean_squared_error(Y_test, predictions)
error
from sklearn.linear_model import SGDRegressor
reg_sgd = SGDRegressor(max_iter=1000, tol=1e-3)
_ = reg_sgd.fit(X_train, Y_train)
predicitons_sgd = reg_sgd.predict(X_test)
error_sgd = mean_squared_error(Y_test, predicitons_sgd)
error_sgd
