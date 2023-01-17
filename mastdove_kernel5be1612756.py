# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras import losses, models, optimizers

from keras.models import Sequential

from keras.layers import (Dense, Dropout, Activation, Flatten) 

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.datasets import load_boston 

from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import ElasticNet, Lasso, Ridge

from matplotlib import pyplot as plt

import numpy as np

import pandas as pd

from geopy import distance





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



data = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", sep=',', nrows=5000, parse_dates=["pickup_datetime"])

is_null = pd.isnull(data).sum()

print(is_null)



print('Old size: %d' % len(data))

data = data.dropna(how = 'any', axis = 'rows')

data = data[data.fare_amount>=0]

print('New size: %d' % len(data))



pd.set_option('display.expand_frame_repr', False) 

print(data.head())

print(data.corr())
# print((data.distance_miles>8000).sum())

print((data.pickup_latitude>90).sum())

print((data.pickup_latitude<-90).sum())

print((data.pickup_longitude>90).sum())

print((data.pickup_longitude<-90).sum())



print((data.dropoff_latitude>90).sum())

print((data.dropoff_latitude<-90).sum())

print((data.dropoff_longitude>90).sum())

print((data.dropoff_longitude<-90).sum())



# print(data.iloc[5686])



data_n = data.drop(data[(data.pickup_latitude>90) | (data.pickup_latitude<-90) 

| (data.pickup_longitude>90) | (data.pickup_longitude<-90) | (data.dropoff_latitude>90) 

| (data.dropoff_latitude<-90) | (data.dropoff_longitude>90) | (data.dropoff_longitude<-90) 

].index)



# | (data.pickup_latitude==0) | (data.pickup_longitude==0)  | (data.dropoff_latitude==0)           

# | (data.dropoff_longitude==0)



data_n = data_n[(data_n.fare_amount != 0) | (data_n.fare_amount != 0.00000)]

data_n = data_n[(data_n.pickup_latitude != 0)]

data_n = data_n[(data_n.pickup_longitude != 0)]

data_n = data_n[(data_n.dropoff_longitude != 0)]

data_n = data_n[(data_n.dropoff_latitude != 0)]

data_n = data_n[(data_n.dropoff_latitude != data_n.pickup_latitude) & (data_n.dropoff_longitude != data_n.pickup_longitude) ]

# df = df[df.line_race != 0]







data_n = data_n.reset_index(drop=True)

# data_n = data[(data.pickup_latitude<90) & (data.pickup_latitude>-90) 

# & (data.pickup_longitude<90) & (data.pickup_longitude>-90) & (data.dropoff_latitude<90) 

# & (data.dropoff_latitude>-90) & (data.dropoff_longitude<90) & (data.dropoff_longitude>-90)]



print((data_n.fare_amount<0).sum())

print((data_n.fare_amount==0).sum())

print((data_n.pickup_latitude==0).sum())

print((data_n.pickup_longitude==0).sum())

print((data_n.dropoff_latitude==0).sum())

print((data_n.dropoff_longitude==0).sum())



print("----------------")



# print((data_n.pickup_latitude>90).sum())

# print((data_n.pickup_latitude<-90).sum())

# print((data_n.pickup_longitude>90).sum())

# print((data_n.pickup_longitude<-90).sum())



# print((data_n.dropoff_latitude>90).sum())

# print((data_n.dropoff_latitude<-90).sum())

# print((data_n.dropoff_longitude>90).sum())

# print((data_n.dropoff_longitude<-90).sum())



print("----------------")



print((data_n.pickup_latitude<90).sum())

print((data_n.pickup_latitude>-90).sum())

print((data_n.pickup_longitude<90).sum())

print((data_n.pickup_longitude>-90).sum())



print((data_n.dropoff_latitude<90).sum())

print((data_n.dropoff_latitude>-90).sum())

print((data_n.dropoff_longitude<90).sum())

print((data_n.dropoff_longitude>-90).sum())

# data = [(data.pickup_latitude<90) & (data.pickup_latitude>-90)]

# data[data.pickup_latitude>90]

# data[data.pickup_latitude<-90]

# data[data.pickup_longitude>90]

# data[data.pickup_longitude<-90]



# data[data.dropoff_latitude>90]

# data[data.dropoff_latitude<-90]

# data[data.dropoff_longitude>90]

# data[data.dropoff_longitude<-90]



print('New size: %d' % len(data_n))

# print(data_n.iloc[5686])
distance_miles = []

for i in range(len(data_n.pickup_latitude)):

    distance_miles.append(distance.distance((data_n.pickup_latitude[i], data_n.pickup_longitude[i]), (data_n.dropoff_latitude[i], data_n.dropoff_longitude[i])).km)

data_n['distance_miles'] = distance_miles 

# data_n = data_n[(data_n.distance_miles != 0)]

# data_n = data_n.reset_index(drop=True)

# data['distance_miles'] = mpu.haversine_distance((data.pickup_latitude, data.pickup_longitude),(data.dropoff_latitude, data.dropoff_longitude)).km 

print(data_n.head())

fig=plt.figure()

ax1=fig.add_subplot(1,1,1)

ax1.scatter(data_n.distance_miles,data_n.fare_amount,  color='r', alpha=0.2)

is_null1 = pd.isnull(data_n.distance_miles).sum()

print("--------------")

print(is_null1)



print(data_n.pickup_datetime)
# Any results you write to the current directory are saved as output.

# print(data_n.distance_miles)

# print(data_n.fare_amount[11])

# print(data_n.iloc[26])

data_n.distance_miles.describe()

data_n.fare_amount.describe()
lr = LinearRegression()

x = data_n.distance_miles

y = data_n.fare_amount



print('New size 2: %d' % len(data_n))

# x = x.reshape(length, 1)

# y = y.reshape(length, 1)

x = np.transpose(np.atleast_2d(x))

lr.fit(x,y)

y_pred = lr.predict(x)

mse_lin_rm = mean_squared_error(y, y_pred)

rmse_lin_rm = np.sqrt(mse_lin_rm)

r2_lin_rm = r2_score(y, y_pred) 



# x = x.values.reshape(len(data_n), 1)

# y = y.values.reshape(len(data_n), 1)

# lr.fit(x, y)



# plt.scatter(x, y,  color='black')

# plt.plot(y, lr.predict(y), color='blue', linewidth=3)

# plt.xticks(())

# plt.yticks(())

# plt.show()

kf = KFold(n_splits=5, random_state=None, shuffle=False)

mse_lin_rm_kf = []

r2_lin_rm_kf = []  

for train_index, test_index in kf.split(x):

    lr.fit(x[train_index],y[train_index])

    mse_lin_rm_kf.append(mean_squared_error(y[test_index], lr.predict(x[test_index])))

    r2_lin_rm_kf.append(r2_score(y[test_index], lr.predict(x[test_index])))

print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(mse_lin_rm_kf), np.std(mse_lin_rm_kf) * 2))

print("Mean R^2: %0.2f" % (np.mean(r2_lin_rm_kf)))



fig=plt.figure(figsize=(11, 8))

ax5=fig.add_subplot(1,1,1)

# ax5.set_xscale('log')



ax5.scatter(data_n.distance_miles,data_n.fare_amount, color='r', alpha=0.2, norm=0.3)

ax5.plot(x,y_pred)

#Назва діаграми

ax5.set_title('Linear Regression ')

ax5.set_xlabel('distance ')

ax5.set_ylabel('fare $USD')

# fig = plt.figure(figsize=(11, 8))

# ax = fig.add_subplot(1, 1, 1)

# ax.set_xscale('log')

# plt.scatter(count, loss,s=area,c=colors,alpha=.5)
# Multiple regression





fig3=plt.figure(figsize=(11, 8))

ax5=fig3.add_subplot(1,1,1)

ax5.scatter(data_n.passenger_count, data_n.fare_amount, color='red')

ax5.set_xlabel('passenger number')

ax5.set_ylabel('fare ')

# ax5.title('datetime vs passenger number')

# ax5.xlabel('passenger number')

# ax5.ylabel('fare')

data_n['year'] = data_n.pickup_datetime.apply(lambda t: t.year)

data_n['weekday'] = data_n.pickup_datetime.apply(lambda t: t.weekday())

data_n['hour'] = data_n.pickup_datetime.apply(lambda t: t.hour)



fig2=plt.figure(figsize=(11, 8))

ax5=fig2.add_subplot(1,1,1)

ax5.scatter(data_n.weekday, data_n.fare_amount, color='red')

ax5.set_xlabel('weekday')

ax5.set_ylabel('fare ')

# ax5.title('datetime vs passenger number')

# ax5.xlabel('passenger number')

# ax5.ylabel('fare')

fig2=plt.figure(figsize=(11, 8))

ax5=fig2.add_subplot(1,1,1)

ax5.scatter(data_n.hour, data_n.fare_amount, color='red')

ax5.set_xlabel('hour')

ax5.set_ylabel('fare ')
# Множинна лінійна регресія

print(data_n.iloc[972])



print(data_n.iloc[969])



x_mul = data_n[['passenger_count','distance_miles']]

y = data_n.fare_amount



X_train, X_test, y_train, y_test = train_test_split(x_mul, y, 

                                                    test_size=0.4, random_state=0)    



lr_mul = LinearRegression()

lr_mul.fit(x_mul,y)

p = lr_mul.predict(x_mul)

mse_lin_mul = mean_squared_error(y, p)

fig=plt.figure()

mulreg=fig.add_subplot(1,1,1)

mulreg.scatter(p, y, color='r', alpha=0.2)

mulreg.set_title('multiLinear Regression ')

# ПРОВЕРКА

passengers = 3

print('\nPredicted fare by multiple regression: ', lr_mul.predict([[passengers,distance]]))





lr_mul = LinearRegression()

lr_mul.fit(X_train, y_train)

y_pred = lr_mul.predict(X_test)

mse_lin_mul = mean_squared_error(y_test, y_pred)

r2_mul = r2_score(y_pred, y_test) 

print('r2_linear', r2_mul)



kf = KFold(n_splits=5, random_state=None, shuffle=False)

mse_lin_mul_kf = []

r2_lin_mul_kf = []  

for train_index, test_index in kf.split(x_mul):

    lr_mul.fit(x_mul[train_index],y[train_index])

    mse_lin_mul_kf.append(mean_squared_error(y[test_index], lr_mul.predict(x_mul[test_index])))

    r2_lin_mul_kf.append(r2_score(y[test_index], lr_mul.predict(x_mul[test_index])))

print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(mse_lin_mul_kf), np.std(mse_lin_mul_kf) * 2))

# print("Mean R^2: %0.2f" % (np.mean(r2_lin_mul_kf)))
### MLPRegressor



mlpReg = MLPRegressor(hidden_layer_sizes=(600, ), activation='tanh', 

                      solver='adam', alpha=0.0001, batch_size='auto', 

                      learning_rate='adaptive', learning_rate_init=0.001, 

                      power_t=0.5, max_iter=10000, shuffle=True, random_state=None, 

                      tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 

                      nesterovs_momentum=True, early_stopping=False, 

                      validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 

                      epsilon=1e-08)



mlpReg = mlpReg.fit(X_train, y_train)

y_pred = mlpReg.predict(X_test)

r2_mlp = r2_score(y_pred, y_test) 

r2_mlp = mlpReg.score(X_test, y_test)



# print(X_test.iloc[5])

print('\nPredicted fare: ', mlpReg.predict([[passengers,distance]]))



                      



# kf = KFold(n_splits=5, random_state=None, shuffle=False)

# mse_mlp_kf = []

# r2_mlp_kf = []  

# for train_index, test_index in kf.split(x_mul):

#     mlpReg.fit(x_mul[train_index],y[train_index])

#     mse_mlp_kf.append(mean_squared_error(y[test_index], mlpReg.predict(x_mul[test_index])))

#     r2_mlp_kf.append(r2_score(y[test_index], mlpReg.predict(x_mul[test_index])))

# print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(mse_mlp_kf), np.std(mse_mlp_kf) * 2))

# print("Mean R^2: %0.2f" % (np.mean(r2_mlp_kf)))
### Deep Neural Net with Keras

kernel_initializer='lecun_uniform'

bias_initializer='zeros'

kernel_regularizer=None

activation = "tanh"

nb_epoch = 1000 # Кількість епох навчання

alpha_zero = 0.001 # Коефіцієнт швидкості навчання

batch_size = 64

model = Sequential()

lpReg = mlpReg.fit(X_train, y_train)

y_pred = mlpReg.predict(X_test)

r2_mlp = r2_score(y_pred, y_test)

r2_mlp = mlpReg.score(X_test, y_test)

############ Додавання повнозв'язного шару 

model.add(Dense(20, input_dim = 13 , activation = activation))

model.add(Dense(15, activation = activation))

model.add(Dense(10, activation = activation))

model.add(Dense(5, activation = activation))

model.add(Dense(1,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer, activation = activation))

############ Компіляція моделі

optimizer = optimizers.Nadam(lr=alpha_zero, beta_1=0.9, beta_2=0.999,epsilon=None, schedule_decay=0.004)

model.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ["accuracy"])

#history = model.fit(X_train, y_train, batch_size = batch_size,epochs = nb_epoch, verbose=2, validation_data = (X_test, y_test))

#score = model.evaluate(X_test, y_test,verbose = 0)

# #print("test score: %f" % score[0])

# #print("test accuracy: %f" % score[1])

model.summary()

#y_pred = model.predict(X_test)

r2_dnn = r2_score(y_pred, y_test)

#mse_dnn = score[0]

print('r2_dnn',r2_dnn)

#print('mse_dnn', mse_dnn)
