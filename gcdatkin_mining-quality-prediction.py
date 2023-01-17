import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import re

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression
data = pd.read_csv('../input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv')
data
data.info()
data.isna().sum()
for column in data.columns:

    data[column] = data[column].apply(lambda x: x.replace(',', '.'))
data
data['date'] = data['date'].apply(lambda x: re.search('[0-9]*-[0-9]*', x).group(0))



data['Year'] = data['date'].apply(lambda x: re.search('^[^-]*', x).group(0))

data['Month'] = data['date'].apply(lambda x: re.search('[^-]*$', x).group(0))



data = data.drop('date', axis=1)
data = data.astype(np.float)
data
plt.figure(figsize=(12, 10))

sns.heatmap(data.corr(), vmin=-1.0, vmax=1.0)

plt.show()
data['Year'].unique()
data = data.drop('Year', axis=1)
data
target = '% Silica Concentrate'



y = data[target]



X_n = data.drop([target, '% Iron Concentrate'], axis=1)

X_i = data.drop(target, axis=1)
scaler = StandardScaler()



X_n = scaler.fit_transform(X_n)

X_i = scaler.fit_transform(X_i)
X_n_train, X_n_test, y_n_train, y_n_test = train_test_split(X_n, y, train_size=0.7)

X_i_train, X_i_test, y_i_train, y_i_test = train_test_split(X_i, y, train_size=0.7)
model_n = LinearRegression()

model_i = LinearRegression()
model_n.fit(X_n_train, y_n_train)

print("Model without iron R^2 Score:", model_n.score(X_n_test, y_n_test))
model_i.fit(X_i_train, y_i_train)

print("Model with iron R^2 Score:", model_i.score(X_i_test, y_i_test))