import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from scipy.stats import  norm

from sklearn import preprocessing

df_original = pd.read_csv('../input/train.csv')

df1 = df_original.copy()
df1['parking_price'] = df1['parking_price'].fillna(0)

df1['parking_area'] = df1['parking_area'].fillna(0)

df1['txn_floor'] = df1['txn_floor'].fillna(0)

df1['village_income_median'] = df1['village_income_median'].fillna(0)



# df1['village_income_median'] = df1['village_income_median'].fillna( df1['village_income_median'].mean())
df1['total_price'] = np.log(df1['total_price'])

df1['total_price'].describe()
df1['nf1'] = np.log(df1.VI_10000 * df1.master_rate)

df1['nf2'] = np.log(df1.X_10000 * df1.master_rate)

df1['nf3'] = np.log(df1.III_10000 * df1.master_rate)

df1['nf4'] = np.log(df1.VIII_10000 * df1.master_rate)

df1['nf5'] = np.log(df1.master_rate * df1.V_10000)
df1_corrmat = df1.corr()
k  = 150

cols = df1_corrmat.nlargest(k,'total_price')['total_price'].index

print(cols)
from sklearn.model_selection import train_test_split

import xgboost

from sklearn import linear_model, svm, gaussian_process

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb



cols = cols.drop('total_price')
df1[cols] = preprocessing.scale(df1[cols])
x = df1[cols].values

y = df1['total_price'].values

# x_scaled = preprocessing.StandardScaler().fit_transform(x)

# y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))

X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

import tensorflow as tf



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Initialising the ANN

model = Sequential()

# Adding the input layer and the first hidden layer

model.add(Dense(70,kernel_initializer='normal',activation = 'relu', input_dim = 149))

model.add(Dense(units = 70,kernel_initializer='normal', activation = 'tanh'))

model.add(Dense(units = 70,kernel_initializer='normal', activation = 'relu'))

model.add(Dense(units = 140, kernel_initializer='normal',activation = 'tanh'))

model.add(Dense(units = 140,kernel_initializer='normal', activation = 'relu'))

model.add(Dense(units = 280,kernel_initializer='normal', activation = 'tanh'))

model.add(Dense(units = 280,kernel_initializer='normal', activation = 'relu'))

model.add(Dense(units = 560,kernel_initializer='normal', activation = 'tanh'))

model.add(Dense(units = 560,kernel_initializer='normal', activation = 'relu'))



# Adding the output layer

model.add(Dense(units = 1))

model.compile(optimizer = 'adam',loss = 'mean_squared_error')



model.fit(X_train, y_train, batch_size = 80, epochs = 200)

# 10 0.1128
y_pred = model.predict(X_test)

plt.plot(y_test, color = 'red', label = 'Real data')

plt.plot(y_pred, color = 'blue', label = 'Predicted data')

plt.title('Prediction')

plt.legend()

plt.show()
df_test_original = pd.read_csv('../input/test.csv')

df1_test = df_test_original.copy()
df1_test['nf1'] = np.log(df1_test.VI_10000 * df1_test.master_rate)

df1_test['nf2'] = np.log(df1_test.X_10000 * df1_test.master_rate)

df1_test['nf3'] = np.log(df1_test.III_10000 * df1_test.master_rate)

df1_test['nf4'] = np.log(df1_test.VIII_10000 * df1_test.master_rate)

df1_test['nf5'] = np.log(df1_test.master_rate * df1_test.V_10000)



df1_test['parking_price'] = df1_test['parking_price'].fillna(0)

df1_test['parking_area'] = df1_test['parking_area'].fillna(0)

df1_test['txn_floor'] = df1_test['txn_floor'].fillna(0)

df1_test['village_income_median'] = df1_test['village_income_median'].fillna(0)



# df1_test['village_income_median'] = df1_test['village_income_median'].fillna( df1_test['village_income_median'].mean())
data_train_test_X = df1_test[cols].values

# data_train_test_Y = final_clt.predict(data_train_test_X)

data_train_test_Y = model.predict(data_train_test_X)



data_train_test_prediction = pd.DataFrame(data_train_test_Y, columns=['total_price'])

data_train_test2 = df1_test.reset_index()

final_result = pd.concat([data_train_test2, data_train_test_prediction],axis = 1)

final_result = final_result.set_index('building_id')

# final_parking_area_result = final_parking_area_result['total_price']

# final_parking_area_result

max_price = max(final_result['total_price'].values)

print('max_price: ' + str(max_price))

min_price = min(final_result['total_price'].values)

print('min_price: ' + str(min_price))

final_result = final_result[['total_price']]

final_result
final_result['total_price'] = np.exp(final_result['total_price'])

max_price = max(final_result['total_price'].values)

print('max_price: ' + str(max_price))

min_price = min(final_result['total_price'].values)

print('min_price: ' + str(min_price))
final_result.to_csv('./answer.csv',index=True)
# import sys

# np.set_printoptions(threshold=sys.maxsize)
# final_result['total_price'].values