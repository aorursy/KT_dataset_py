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
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_log_error



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.layers import GRU

from keras.initializers import random_uniform

from keras.optimizers import Adagrad

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



import tensorflow as tf

import datetime

import matplotlib.pyplot as plt

plt.style.use('ggplot')

font = {'family' : 'meiryo'}

plt.rc('font', **font)
import random as rn

import os

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)

session_conf =  tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

tf.random.set_seed(1234)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)
train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

train_df = train_df.fillna("No State")

train_df
test_rate = 0.1

time_series_len = 18

train_date_count = len(set(train_df["Date"]))



X, Y = [],[]



scaler = StandardScaler()

train_df["ConfirmedCases_std"] = scaler.fit_transform(train_df["ConfirmedCases"].values.reshape(len(train_df["ConfirmedCases"].values),1))



#Formatting the train data for a time series model

for state,country in train_df.groupby(["Province_State","Country_Region"]).sum().index:

    df = train_df[(train_df["Country_Region"] == country) & (train_df["Province_State"] == state)]

    

    #Areas with zero patients cannot be predicted ⇒ Artificially predicted to be zero

    if df["ConfirmedCases"].sum() != 0:

        for i in range(len(df) - time_series_len):

            X.append(df[['ConfirmedCases_std']].iloc[i:(i+time_series_len)].values)

            Y.append(df[['ConfirmedCases_std']].iloc[i+time_series_len].values)



X=np.array(X)

Y=np.array(Y)

    

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_rate, shuffle = True ,random_state = 0)
confirmedCases_std_min = train_df["ConfirmedCases_std"].min()
def huber_loss(y_true, y_pred, clip_delta=1.0):

  error = y_true - y_pred

  cond  = tf.keras.backend.abs(error) < clip_delta



  squared_loss = 0.5 * tf.keras.backend.square(error)

  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)



  return tf.where(cond, squared_loss, linear_loss)



def huber_loss_mean(y_true, y_pred, clip_delta=1.0):

  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))
epochs_num = 20

n_in = 1



model = Sequential()

model.add(GRU(100,

               batch_input_shape=(None, time_series_len, n_in),

               kernel_initializer=random_uniform(seed=1),

               return_sequences=False

             ))

model.add(Dense(50))

model.add(Dropout(0.15))

model.add(Dense(n_in, kernel_initializer=random_uniform(seed=1)))

model.add(Activation("linear"))



opt = Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4)

model.compile(loss = huber_loss_mean, optimizer=opt)
callbacks = [ReduceLROnPlateau(monitor='loss', patience=4, verbose=1, factor=0.6),

             EarlyStopping(monitor='loss', patience=10)]



hist = model.fit(X_train, Y_train, batch_size=20, epochs=epochs_num,

                 callbacks=callbacks,shuffle=False)
predicted_std = model.predict(X_test)

result_std= pd.DataFrame(predicted_std)

result_std.columns = ['predict']

result_std['actual'] = Y_test

result_std.plot(figsize=(25,6))

plt.show()
loss = hist.history['loss']

epochs = len(loss)

fig = plt.figure()

plt.plot(range(epochs), loss, marker='.', label='loss(training data)')

plt.show()
predicted = scaler.inverse_transform(predicted_std)

Y_test2 = scaler.inverse_transform(Y_test)
np.sqrt(mean_squared_log_error(predicted, Y_test2))
result= pd.DataFrame(predicted)

result.columns = ['predict']

result['actual'] = Y_test2

result.plot(figsize=(25,6))

plt.show()
test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")
temp = (datetime.datetime.strptime("2020-04-01", '%Y-%m-%d') - datetime.timedelta(days=time_series_len)).strftime('%Y-%m-%d')

test_df = train_df[train_df["Date"] > temp]
check_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv").query("Date>'2020-04-01'and Date<='2020-04-14'")

check_df["ConfirmedCases_std"] = scaler.transform(check_df["ConfirmedCases"].values.reshape(len(check_df["ConfirmedCases"].values),1))
confirmedCases_pred = []

for i in range(0,313*time_series_len,time_series_len):

    temp_array = np.array(test_df["ConfirmedCases_std"][i:i+time_series_len])

    for j in range(43):

        if j<13:

            temp_array = np.append(temp_array,np.array(check_df["ConfirmedCases_std"])[int(i*13/time_series_len)+j])

        elif np.array(test_df["ConfirmedCases"][i:i+time_series_len]).sum() == 0:

            temp_array = np.append(temp_array,temp_array[-1])

        else:

            temp_array = np.append(temp_array,model.predict(temp_array[-time_series_len:].reshape(1,time_series_len,1)))

    confirmedCases_pred.append(temp_array[-43:])
submission["ConfirmedCases"] = np.abs(scaler.inverse_transform(np.array(confirmedCases_pred).reshape(313*43)))

submission["ConfirmedCases_std"] = np.array(confirmedCases_pred).reshape(313*43)

submission
submission.to_csv('./submission_c.csv')

submission.to_csv('..\output\kaggle\working\submission_c.csv')
test_rate = 0.1

time_series_len = 16

train_date_count = len(set(train_df["Date"]))



X, Y = [],[]



scaler = StandardScaler()

train_df["Fatalities_std"] = scaler.fit_transform(train_df["Fatalities"].values.reshape(len(train_df["Fatalities"].values),1))



ss = StandardScaler()

train_df["ConfirmedCases_std"] = ss.fit_transform(train_df["ConfirmedCases"].values.reshape(len(train_df["ConfirmedCases"].values),1))



#Formatting the train data for a time series model

for state,country in train_df.groupby(["Province_State","Country_Region"]).sum().index:

    df = train_df[(train_df["Country_Region"] == country) & (train_df["Province_State"] == state)]

    

    #Areas with zero patients cannot be predicted ⇒ Artificially predicted to be zero

    if df["Fatalities"].sum() != 0 or df["ConfirmedCases"].sum() != 0:

        for i in range(len(df) - time_series_len):

            X.append(df[['Fatalities_std','ConfirmedCases_std']].iloc[i:(i+time_series_len)].values)

            Y.append(df[['Fatalities_std']].iloc[i+time_series_len].values)



X=np.array(X)

Y=np.array(Y)

    

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_rate, shuffle = True ,random_state = 0)
fatalities_std_min = train_df["Fatalities_std"].min()
epochs_num = 20

n_in = 2



model = Sequential()

model.add(GRU(100,

               batch_input_shape=(None, time_series_len, n_in),

               kernel_initializer=random_uniform(seed=1),

               return_sequences=False))

model.add(Dense(50))

model.add(Dropout(0.19))

model.add(Dense(1, kernel_initializer=random_uniform(seed=1)))

model.add(Activation("linear"))



opt = Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4)

model.compile(loss = huber_loss_mean, optimizer=opt)
callbacks = [ReduceLROnPlateau(monitor='loss', patience=4, verbose=1, factor=0.6),

             EarlyStopping(monitor='loss', patience=10)]

hist = model.fit(X_train, Y_train, batch_size=16, epochs=epochs_num,

                 callbacks=callbacks,shuffle=False)
predicted_std = model.predict(X_test)

result_std= pd.DataFrame(predicted_std)

result_std.columns = ['predict']

result_std['actual'] = Y_test
result_std.plot(figsize=(25,6))

plt.show()
loss = hist.history['loss']

epochs = len(loss)

fig = plt.figure()

plt.plot(range(epochs), loss, marker='.', label='loss(training data)')

plt.show()
predicted = scaler.inverse_transform(predicted_std)

Y_test = scaler.inverse_transform(Y_test)
X_test_ = scaler.inverse_transform(X_test)
np.sqrt(mean_squared_log_error(predicted, Y_test))
temp = (datetime.datetime.strptime("2020-04-01", '%Y-%m-%d') - datetime.timedelta(days=time_series_len)).strftime('%Y-%m-%d')

test_df = train_df[train_df["Date"] > temp]
check_df["Fatalities_std"] = scaler.transform(check_df["Fatalities"].values.reshape(len(check_df["Fatalities"].values),1))

check_df
fatalities_pred = []

for i in range(0,313*time_series_len,time_series_len):

    temp_array = np.array(test_df[["Fatalities_std","ConfirmedCases_std"]][i:i+time_series_len])

    for j in range(43):

        if j<13:

            temp_array = np.append(temp_array,np.append(np.array(check_df["Fatalities_std"])[int(i*13/time_series_len)+j],np.array(check_df["ConfirmedCases_std"])[int(i*13/time_series_len)+j]).reshape(1,2),axis=0)

        elif np.array(test_df[["Fatalities","ConfirmedCases"]][i:i+time_series_len]).sum() == 0:

            temp_array = np.append(temp_array,np.array(temp_array[-1]).reshape(1,2),axis=0)

        else:

            temp_array = np.append(temp_array,np.append(model.predict(temp_array[-time_series_len:].reshape(1,time_series_len,2)),submission["ConfirmedCases_std"][i/time_series_len*43+j]).reshape(1,2),axis=0)

    fatalities_pred.append(temp_array[-43:])
submission["Fatalities"] = np.abs(scaler.inverse_transform([i[0] for i in np.array(fatalities_pred).reshape(313*43,2)]))

submission
submission[["ConfirmedCases","Fatalities"]] = submission[["ConfirmedCases","Fatalities"]].round().astype(int)

submission
submission = submission.drop("ConfirmedCases_std",axis=1)
submission = submission.set_index('ForecastId')
for i in range(313):

    for j in range(2,44):

        if submission["ConfirmedCases"][i*43+j] < submission["ConfirmedCases"][i*43+j-1]:

            submission["ConfirmedCases"][i*43+j] = submission["ConfirmedCases"][i*43+j-1]

        if submission["Fatalities"][i*43+j] < submission["Fatalities"][i*43+j-1]:

            submission["Fatalities"][i*43+j] = submission["Fatalities"][i*43+j-1]
submission.to_csv('submission.csv')