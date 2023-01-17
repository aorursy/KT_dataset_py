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

from keras.layers.core import Dense, Activation

from keras.layers.recurrent import GRU

from keras.optimizers import Adagrad

from keras.callbacks import EarlyStopping

from keras import backend as K



import datetime

import matplotlib.pyplot as plt

plt.style.use('ggplot')

font = {'family' : 'meiryo'}

plt.rc('font', **font)
train_df = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
train_df = train_df[train_df["Date"] < "2020-03-27"]

train_df = train_df.fillna("No State")
test_rate = 0.05

maxlen = 13

train_date_count = len(set(train_df["Date"]))



X, Y = [],[]



scaler = StandardScaler()

train_df["ConfirmedCases_std"] = scaler.fit_transform(train_df["ConfirmedCases"].values.reshape(len(train_df["ConfirmedCases"].values),1))

confirmedCase_std_min = train_df["ConfirmedCases_std"].min()

train_df["ConfirmedCases_std"] = train_df["ConfirmedCases_std"] + abs(confirmedCase_std_min)



#時系列モデル用に学習データを整形する

for state,country in train_df.groupby(["Province_State","Country_Region"]).sum().index:

    df = train_df[(train_df["Country_Region"] == country) & (train_df["Province_State"] == state)]

    

    #患者が0人の地域は予想不可⇒人為的に0で予想する

    if df["ConfirmedCases"].sum() != 0:

        for i in range(len(df) - maxlen):

            

            #時系列データの患者が0人の場合は除外

            #if df[['ConfirmedCases']].iloc[i+maxlen].values != 0:

            X.append(df[['ConfirmedCases_std']].iloc[i:(i+maxlen)].values)

            Y.append(df[['ConfirmedCases_std']].iloc[i+maxlen].values)



X=np.array(X)

Y=np.array(Y)

    

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_rate, shuffle = True ,random_state = 0)
import tensorflow as tf



def huber_loss(y_true, y_pred, clip_delta=1.0):

  error = y_true - y_pred

  cond  = tf.keras.backend.abs(error) < clip_delta



  squared_loss = 0.5 * tf.keras.backend.square(error)

  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)



  return tf.where(cond, squared_loss, linear_loss)



def huber_loss_mean(y_true, y_pred, clip_delta=1.0):

  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))



def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
epochs_num = 10

n_hidden = 200

n_in = 1

    

model = Sequential()

model.add(GRU(n_hidden,

               batch_input_shape=(None, maxlen, n_in),

               kernel_initializer='random_uniform',

               return_sequences=False))

model.add(Dense(n_in, kernel_initializer='random_uniform'))

model.add(Activation("linear"))



opt = Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4)

#model.compile(loss = "mean_squared_error", optimizer=opt)

model.compile(loss = huber_loss_mean, optimizer=opt)
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)



hist = model.fit(X_train, Y_train, batch_size=10, epochs=epochs_num,

                 callbacks=[early_stopping],shuffle=False)
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
predicted = scaler.inverse_transform(predicted_std - abs(confirmedCase_std_min))

Y_test = scaler.inverse_transform(Y_test - abs(confirmedCase_std_min))
np.sqrt(mean_squared_log_error(predicted, Y_test))
result= pd.DataFrame(predicted)

result.columns = ['predict']

result['actual'] = Y_test

result.plot(figsize=(25,6))

plt.show()
test_df = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

test_df
submission_c = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")
temp = (datetime.datetime.strptime("2020-03-25", '%Y-%m-%d') - datetime.timedelta(days=maxlen)).strftime('%Y-%m-%d')

test_df = train_df[train_df["Date"] > temp]
check_df = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv").query("Date>'2020-03-25'and Date<='2020-04-07'")

check_df["ConfirmedCases_std"] = scaler.transform(check_df["ConfirmedCases"].values.reshape(len(check_df["ConfirmedCases"].values),1)) + abs(confirmedCase_std_min)
confirmedCases_pred = []

for i in range(0,306*maxlen,maxlen):

    temp_array = np.array(test_df["ConfirmedCases_std"][i:i+maxlen])

    for j in range(43):

        if j<13:

            temp_array = np.append(temp_array,np.array(check_df["ConfirmedCases_std"])[int(i*13/maxlen)+j])

        elif np.array(test_df["ConfirmedCases"][i:i+maxlen]).sum() == 0:

            temp_array = np.append(temp_array,temp_array[-1])

        else:

            pred = model.predict(temp_array[-maxlen:].reshape(1,maxlen,1))

            pred = np.where(float(pred) < temp_array[-1],temp_array[-1],pred)

            temp_array = np.append(temp_array,pred)

    confirmedCases_pred.append(temp_array[-43:])
submission_c["ConfirmedCases"] = np.abs(scaler.inverse_transform(np.array(confirmedCases_pred).reshape(306*43) - abs(confirmedCase_std_min)))

submission_c["ConfirmedCases_std"] = np.array(confirmedCases_pred).reshape(306*43)

submission_c
submission_c.to_csv('./submission_c.csv')

submission_c.to_csv('..\output\kaggle\working\submission_c.csv')
test_rate = 0.05

maxlen = 13

train_date_count = len(set(train_df["Date"]))



X, Y = [],[]



scaler = StandardScaler()

train_df["Fatalities_std"] = scaler.fit_transform(train_df["Fatalities"].values.reshape(len(train_df["Fatalities"].values),1))

fatalities_std_min = train_df["Fatalities_std"].min()

train_df["Fatalities_std"] = train_df["Fatalities_std"] + abs(fatalities_std_min)



ss = StandardScaler()

train_df["ConfirmedCases_std"] = ss.fit_transform(train_df["ConfirmedCases"].values.reshape(len(train_df["ConfirmedCases"].values),1))

confirmedCase_std_min = train_df["ConfirmedCases_std"].min()

train_df["ConfirmedCases_std"] = train_df["ConfirmedCases_std"] + abs(confirmedCase_std_min)



#時系列モデル用に学習データを整形する

for state,country in train_df.groupby(["Province_State","Country_Region"]).sum().index:

    df = train_df[(train_df["Country_Region"] == country) & (train_df["Province_State"] == state)]

    

    #患者と重傷者が0人の地域は予想不可

    if df["Fatalities"].sum() != 0 or df["ConfirmedCases"].sum() != 0:

        for i in range(len(df) - maxlen):

            

            #時系列データの患者と重傷者が0人の場合は除外

            #if (df[['ConfirmedCases']].iloc[i+maxlen].values != 0 or df[['Fatalities']].iloc[i+maxlen].values != 0):

            X.append(df[['Fatalities_std','ConfirmedCases_std']].iloc[i:(i+maxlen)].values)

            Y.append(df[['Fatalities_std']].iloc[i+maxlen].values)



X=np.array(X)

Y=np.array(Y)

    

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_rate, shuffle = True ,random_state = 0)
epochs_num = 30

n_hidden = 200

n_in = 2



model = Sequential()

model.add(GRU(n_hidden,

               batch_input_shape=(None, maxlen, n_in),

               kernel_initializer='random_uniform',

               return_sequences=False))

model.add(Dense(1, kernel_initializer='random_uniform'))

model.add(Activation("linear"))



opt = Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4)

#model.compile(loss = "mean_squared_error", optimizer=opt)

model.compile(loss = huber_loss_mean, optimizer=opt)
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)



hist = model.fit(X_train, Y_train, batch_size=8, epochs=epochs_num,

                 callbacks=[early_stopping],shuffle=False)
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
predicted = scaler.inverse_transform(predicted_std - abs(fatalities_std_min))

Y_test = scaler.inverse_transform(Y_test - abs(fatalities_std_min))
X_test_ = scaler.inverse_transform(X_test - abs(fatalities_std_min))
submission_df = submission_c
temp = (datetime.datetime.strptime("2020-03-25", '%Y-%m-%d') - datetime.timedelta(days=maxlen)).strftime('%Y-%m-%d')

test_df = train_df[train_df["Date"] > temp]
submission_df
check_df["Fatalities_std"] = scaler.transform(check_df["Fatalities"].values.reshape(len(check_df["Fatalities"].values),1)) + abs(fatalities_std_min)

check_df
fatalities_pred = []

for i in range(0,306*maxlen,maxlen):

    temp_array = np.array(test_df[["Fatalities_std","ConfirmedCases_std"]][i:i+maxlen])

    for j in range(43):

        if j<13:

            temp_array = np.append(temp_array,np.append(np.array(check_df["Fatalities_std"])[int(i*13/maxlen)+j],np.array(check_df["ConfirmedCases_std"])[int(i*13/maxlen)+j]).reshape(1,2),axis=0)

        elif np.array(test_df[["Fatalities","ConfirmedCases"]][i:i+maxlen]).sum() == 0:

            temp_array = np.append(temp_array,np.array(temp_array[-1]).reshape(1,2),axis=0)

        else:

            pred = model.predict(temp_array[-maxlen:].reshape(1,maxlen,2))

            pred = np.where(pred<temp_array[-1][0],temp_array[-1][0],pred)

            temp_array = np.append(temp_array,np.append(pred,submission_df["ConfirmedCases_std"][i/maxlen*43+j]).reshape(1,2),axis=0)

    fatalities_pred.append(temp_array[-43:])
submission_df["Fatalities"] = np.abs(scaler.inverse_transform([i[0] - abs(fatalities_std_min) for i in np.array(fatalities_pred).reshape(306*43,2)]))

submission_df
submission_df[["ConfirmedCases","Fatalities"]] = submission_df[["ConfirmedCases","Fatalities"]].round().astype(int)

submission_df
submission_df = submission_df.drop("ConfirmedCases_std",axis=1)
submission_df = submission_df.set_index('ForecastId')
submission_df
submission_df.to_csv('submission.csv')