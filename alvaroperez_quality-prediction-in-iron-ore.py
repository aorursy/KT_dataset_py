import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/MiningProcess_Flotation_Plant_Database.csv",decimal=",",parse_dates=["date"],infer_datetime_format=True).drop_duplicates()
df.head()
df.describe()
df = df.dropna()

df.shape
plt.figure(figsize=(30, 25))

p = sns.heatmap(df.corr(), annot=True)
silic_corr = df.corr()['% Silica Concentrate']

silic_corr = abs(silic_corr).sort_values()

silic_corr
drop_index= silic_corr.index[:5].tolist()+["date"]#+["% Iron Concentrate"]

print (drop_index)
df = df.drop(drop_index, axis=1)
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split

from IPython.display import display

from sklearn import metrics
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_features=None, n_jobs=-1)

model.fit(X_train,Y_train)
predicitons_rdmforest = model.predict(X_test)

error_rdf = mean_squared_error(Y_test, predicitons_rdmforest)

error_rdf
from sklearn import datasets

from sklearn.model_selection import cross_val_predict

from sklearn import linear_model

import matplotlib.pyplot as plt

ax = plt.subplots()
pred_1, pred_2, Y_1, Y_2 = train_test_split(predicitons_rdmforest, Y_test, test_size=0.9, random_state=42)
plt.figure(figsize=(20,20))

plt.scatter(Y_1, pred_1)

plt.plot([Y_1.min(), Y_1.max()], [Y_1.min(), Y_1.max()], 'k--', lw=4)

plt.show()
plt.show(ax)

