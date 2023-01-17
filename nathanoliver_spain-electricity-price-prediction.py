import pandas as pd

import numpy as np

import seaborn as sns

import scipy

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

from math import sqrt

from sklearn.metrics import mean_squared_error

from openpyxl import load_workbook
data_energy = pd.read_csv('../input/energy-consumption-generation-prices-and-weather/energy_dataset.csv')
hour = data_energy.time.str.slice(11, 13)



df = pd.DataFrame(data_energy)



df['hour'] = hour



df['time'] = pd.to_datetime(df['time'], utc=True)
df['hour'] = df['hour'].astype(str)



new_dummies = pd.get_dummies(df['hour'])



df = pd.concat([df,new_dummies],axis='columns')
plt.figure(figsize=(10,4))

gr = sns.distplot(df['price day ahead'], bins=50, label='TSO Prediction')

gr = sns.distplot(df['price actual'], bins=50, label='Actual Price')

gr.set(xlabel="Price of Electricity (€/MWh)", ylabel="Frequency")

gr.set_title('Electricity Price Comparison\nActual Price vs. TSO Prediction')

plt.legend()

plt.show()



p_diff = df['price actual'] - df['price day ahead']



plt.figure(figsize=(10,4))

gr = sns.distplot(p_diff, bins=50, label='Actual - Predicted')

gr.set(xlabel="Price of Electricity (€/MWh)", ylabel="Frequency")

gr.set_title('Electricity Price Difference\nActual Price vs. TSO Prediction')

plt.legend()

plt.show()
X = ['forecast solar day ahead','forecast wind onshore day ahead', 'total load forecast','00','01','02','03','04','05',

     '06', '07', '08', '09','10', '11', '12', '13', '14', '15', '16' ,'17', '18', '19', '20', '21', '22', '23','price day ahead']

y = ['price day ahead', 'price actual']



df = df.dropna(subset=X)

df = df.dropna(subset=y)



for i in df[X]:

    df[i] = pd.to_numeric(df[i])



for i in df[y]:

    df[i] = pd.to_numeric(df[i])
X_train, X_test, y_train, y_test = train_test_split(df[X],df[y],test_size=0.2, random_state=0)



lr_model = LinearRegression()



lr_model.fit(X_train,y_train['price actual'])



print(lr_model.score(X_test,y_test['price actual']))



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
val_predictions = lr_model.predict(X_test)



diff1 = y_test['price actual'] - val_predictions

diff2 = y_test['price actual'] - y_test['price day ahead']



# f, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)



plt.figure(figsize=(10,4))

gr = sns.distplot(diff1, bins=50, label='Actual Price - My Predictions')

gr = sns.distplot(diff2, bins=50, label='Actual Price - TSO Predictions')

gr.set(xlabel="Price Difference (€/MWh)", ylabel="Frequency")

gr.set_title('Price Difference Comparison\nMy Prediction vs. ISO Prediction')

gr.set(xlim=(-60,60))

plt.legend()

plt.show()



plt.figure(figsize=(10,4))

gr = sns.distplot(y_test['price actual'], bins=50, label='Actual Price')

gr = sns.distplot(y_test['price day ahead'], bins=50, label='TSO Prediction')

gr = sns.distplot(val_predictions, bins=50, label='My Prediction')

gr.set(xlabel="Price of Electricity (€/MWh)", ylabel="Frequency")

gr.set_title('Actual Price, My Prediction, TSO Prediction')

gr.set(xlim=(0,None))

plt.legend()

plt.show()
rms_my_pred = sqrt(mean_squared_error(y_test['price actual'], val_predictions))

rms_TSO_pred = sqrt(mean_squared_error(

    y_test['price actual'], y_test['price day ahead']))



mean_abs_error_my_pred = mean_absolute_error(

    y_test['price actual'], val_predictions)

mean_abs_error_TSO_pred = mean_absolute_error(

    y_test['price actual'], y_test['price day ahead'])



print('Multiple Linear Regression Error: ' +

      str(rms_my_pred))

print('TSO Error: ' +

      str(rms_TSO_pred))