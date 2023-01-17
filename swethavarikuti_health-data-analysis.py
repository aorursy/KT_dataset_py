import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/Big_Cities_Health_Data_Inventory.csv')
df.head()
df.info()
df.describe(include='all')
df.drop(columns=['Methods','Notes'],inplace=True)
df['Indicator Category'].value_counts()
groupedvalues=df.groupby('Indicator Category').sum().reset_index()
groupedvalues.head()


plt.figure(figsize=(40,5)) 

sns.set(style="whitegrid")

groupedvalues=df.groupby('Indicator Category').sum().reset_index()

g = sns.barplot(groupedvalues['Indicator Category'],groupedvalues['Value'])

for index, row in groupedvalues.iterrows():

    g.text(row.name,row.Value, round(row.Value,2), color='black', ha="center")

plt.show()


plt.figure(figsize=(50,15))

sns.set(style="whitegrid")

groupedvalues=df.groupby('Place').sum().reset_index()

g = sns.barplot(groupedvalues['Place'],groupedvalues['Value'])

for index, row in groupedvalues.iterrows():

    g.text(row.name,row.Value, round(row.Value,2), color='black', ha="center")

g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
df.dropna(inplace=True)
df_encoded = pd.get_dummies(df)
X = df_encoded.drop(columns=['Value'])

y = df_encoded['Value']
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 101)

model = LinearRegression()

model.fit(X_train,y_train)
model.coef_
model.intercept_
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

train_predict = model.predict(X_train)



mae_train = mean_absolute_error(y_train,train_predict)



mse_train = mean_squared_error(y_train,train_predict)



rmse_train = np.sqrt(mse_train)



r2_train = r2_score(y_train,train_predict)



mape_train = mean_absolute_percentage_error(y_train,train_predict)
test_predict = model.predict(X_test)



mae_test = mean_absolute_error(test_predict,y_test)



mse_test = mean_squared_error(test_predict,y_test)



rmse_test = np.sqrt(mean_squared_error(test_predict,y_test))



r2_test = r2_score(y_test,test_predict)



mape_test = mean_absolute_percentage_error(y_test,test_predict)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

print('TRAIN: Mean Absolute Error(MAE): ',mae_train)

print('TRAIN: Mean Squared Error(MSE):',mse_train)

print('TRAIN: Root Mean Squared Error(RMSE):',rmse_train)

print('TRAIN: R square value:',r2_train)

print('TRAIN: Mean Absolute Percentage Error: ',mape_train)

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

print('TEST: Mean Absolute Error(MAE): ',mae_test)

print('TEST: Mean Squared Error(MSE):',mse_test)

print('TEST: Root Mean Squared Error(RMSE):',rmse_test)

print('TEST: R square value:',r2_test)

print('TEST: Mean Absolute Percentage Error: ',mape_test)