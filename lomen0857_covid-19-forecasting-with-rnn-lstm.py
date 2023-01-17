import datetime

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.dates as mdates

import matplotlib.pyplot as plt

plt.style.use('ggplot')



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_log_error



from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.layers.recurrent import LSTM

from keras.optimizers import Adagrad

from keras.callbacks import EarlyStopping
train_df = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

train_df
train_df["Province_State"] = train_df["Province_State"].fillna("No State")
plt.figure(figsize=(25,6))

plt.title("ConfirmedCases")

plt.plot(range(len(train_df["ConfirmedCases"].values)),train_df["ConfirmedCases"].values)

plt.show()



plt.figure(figsize=(25,6))

plt.title("Fatalities")

plt.plot(range(len(train_df["Fatalities"].values)),train_df["Fatalities"].values)

plt.show()
labels = list(set(train_df["Date"].values))

ticks = 5



plt.figure(figsize=(25,6))

plt.title("ConfirmedCases(Japan)")

plt.plot(train_df["Date"][train_df["Country_Region"] == "Japan"].values,train_df[["ConfirmedCases","Fatalities"]][train_df["Country_Region"] == "Japan"].values)

plt.xticks(range(0, len(labels), ticks), labels[::ticks])

plt.show()



plt.figure(figsize=(25,6))

plt.title("Fatalities(Japan)")

plt.plot(train_df["Date"][train_df["Country_Region"] == "Japan"].values,train_df["Fatalities"][train_df["Country_Region"] == "Japan"].values)

plt.xticks(range(0, len(labels), ticks), labels[::ticks])

plt.show()
#Ratio of test data

test_rate = 0.1



#Length of time series data

time_series_len = 20



#Number of dates that can be used as training data(1/22～3/18 = 57!)

train_data_date_count = len(set(train_df["Date"]))
#Preprocessing

ss_c = StandardScaler()

train_df["ConfirmedCases_std"] = ss_c.fit_transform(train_df["ConfirmedCases"].values.reshape(len(train_df["ConfirmedCases"].values),1))



ss_f = StandardScaler()

train_df["Fatalities_std"] = ss_f.fit_transform(train_df["Fatalities"].values.reshape(len(train_df["Fatalities"].values),1))
X, Y_c, Y_f = [],[],[]



for state,country in train_df.groupby(["Province_State","Country_Region"]).sum().index:

    df = train_df[(train_df["Country_Region"] == country) & (train_df["Province_State"] == state)]

    

    if df["ConfirmedCases"].sum() != 0 or df["Fatalities"].sum() != 0:

        

        for i in range(len(df) - time_series_len):

            

            if (df[['ConfirmedCases']].iloc[i+time_series_len-1].values != 0 or df[['Fatalities']].iloc[i+time_series_len-1].values != 0):

                X.append(df[['ConfirmedCases_std','Fatalities_std']].iloc[i:(i+time_series_len)].values)

                Y_c.append(df[['ConfirmedCases_std']].iloc[i+time_series_len].values)

                Y_f.append(df[['Fatalities_std']].iloc[i+time_series_len].values)



X=np.array(X)

Y_f=np.array(Y_f)

Y_c=np.array(Y_c)
print(X.shape)

print(X)
print(Y_c.shape)

print(Y_c)
print(Y_f.shape)

print(Y_f)
#split

X_train, X_test, Y_c_train, Y_c_test = train_test_split(X, Y_c, test_size=test_rate, shuffle = True ,random_state = 0)

X_train, X_test, Y_f_train, Y_f_test = train_test_split(X, Y_f, test_size=test_rate, shuffle = True ,random_state = 0)
#Stores the minimum value after standardization, i.e., the value with 0 standardized.

confirmedCases_std_min = train_df["ConfirmedCases_std"].min()

fatalities_std_min = train_df["Fatalities_std"].min()
#loss function

def huber_loss(y_true, y_pred, clip_delta=1.0):

  error = y_true - y_pred

  cond  = tf.keras.backend.abs(error) < clip_delta



  squared_loss = 0.5 * tf.keras.backend.square(error)

  linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)



  return tf.where(cond, squared_loss, linear_loss)



def huber_loss_mean(y_true, y_pred, clip_delta=1.0):

  return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))
epochs_num = 20

batch_size_num = 10

n_hidden = 300

n_in = 2



model_c = Sequential()

model_c.add(LSTM(n_hidden,

               batch_input_shape=(None, time_series_len, n_in),

               kernel_initializer='random_uniform',

               return_sequences=False))

model_c.add(Dense(1, kernel_initializer='random_uniform'))

model_c.add(Activation("linear"))

opt = Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4)

model_c.compile(loss = huber_loss_mean, optimizer=opt)



model_f = Sequential()

model_f.add(LSTM(n_hidden,

               batch_input_shape=(None, time_series_len, n_in),

               kernel_initializer='random_uniform',

               return_sequences=False))

model_f.add(Dense(1, kernel_initializer='random_uniform'))

model_f.add(Activation("linear"))

opt = Adagrad(lr=0.01, epsilon=1e-08, decay=1e-4)

model_f.compile(loss = huber_loss_mean, optimizer=opt)
early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)



hist_c = model_c.fit(X_train, Y_c_train, batch_size=batch_size_num, epochs=epochs_num,

                 callbacks=[early_stopping],shuffle=False)



hist_f = model_f.fit(X_train, Y_f_train, batch_size=batch_size_num, epochs=epochs_num,

                 callbacks=[early_stopping],shuffle=False)
#predict

predicted_c_std = model_c.predict(X_test)

result_c_std= pd.DataFrame(predicted_c_std)

result_c_std.columns = ['predict']

result_c_std['actual'] = Y_c_test



predicted_f_std = model_f.predict(X_test)

result_f_std= pd.DataFrame(predicted_f_std)

result_f_std.columns = ['predict']

result_f_std['actual'] = Y_f_test
loss_c = hist_c.history["loss"]

epochs = len(loss_c)

plt.figure()

plt.title("loss(ConfirmedCases)")

plt.plot(range(epochs), loss_c, marker=".")

plt.show()



loss_f = hist_f.history["loss"]

epochs = len(loss_f)

plt.figure()

plt.title("loss(Fatalities)")

plt.plot(range(epochs), loss_f, marker=".")

plt.show()
#Confirming the expected results in the standardized state

plt.figure()

result_f_std[:30].plot.bar(title = "ConfirmedCases_std")

plt.show()



plt.figure()

result_c_std[:30].plot.bar(title = "Fatalities_std")

plt.show()
#Evaluate it back from standardization.

predicted_c = ss_c.inverse_transform(predicted_c_std)

Y_c_inv_test = ss_c.inverse_transform(Y_c_test)



predicted_f = ss_f.inverse_transform(predicted_f_std)

Y_f_inv_test = ss_f.inverse_transform(Y_f_test)
#Check the forecast results

result= pd.DataFrame(predicted_c)

result.columns = ['predict']

result['actual'] = Y_c_inv_test

result[:30].plot.bar(title = "ConfirmedCases")

plt.show()



result= pd.DataFrame(predicted_f)

result.columns = ['predict']

result['actual'] = Y_f_inv_test

result[:30].plot.bar(title = "Fatalities")

plt.show()
submission = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

test_date_count = len(set(test_df["Date"]))



region_count = len(train_df.groupby(["Province_State","Country_Region"]))
#Create the time series data needed to predict a date after 3/19

temp = (datetime.datetime.strptime("2020-03-18", '%Y-%m-%d') - datetime.timedelta(days=time_series_len)).strftime('%Y-%m-%d')

test_df = train_df[train_df["Date"] > temp]
f_pred = []

c_pred = []



for i in range(0,region_count*time_series_len,time_series_len):

    temp_array = np.array(test_df[["ConfirmedCases_std","Fatalities_std"]][i:i+time_series_len])

    for j in range(test_date_count):

        if np.array(test_df[["ConfirmedCases","Fatalities"]][i:i+time_series_len]).sum() == 0:

            temp_array = np.append(temp_array,np.append(confirmedCases_std_min,fatalities_std_min).reshape(1,2),axis=0)

        else:

            temp_array = np.append(temp_array,np.append(model_c.predict(temp_array[-time_series_len:].reshape(1,time_series_len,2)),

                                                        model_f.predict(temp_array[-time_series_len:].reshape(1,time_series_len,2))).reshape(1,2),axis=0)

    c_pred.append([i[0] for i in temp_array[-test_date_count:]])

    f_pred.append([i[1] for i in temp_array[-test_date_count:]])
submission["ConfirmedCases"] = np.abs(ss_c.inverse_transform(np.array(c_pred).reshape(region_count*test_date_count)))

submission["ConfirmedCases_std"] = np.array(c_pred).reshape(region_count*test_date_count)



submission["Fatalities"] = np.abs(ss_f.inverse_transform(np.array(f_pred).reshape(region_count*test_date_count)))

submission["Fatalities_std"] = np.array(f_pred).reshape(region_count*test_date_count)



submission
submission[["ConfirmedCases","Fatalities"]] = submission[["ConfirmedCases","Fatalities"]].round().astype(int)

submission = submission.drop(["ConfirmedCases_std","Fatalities_std"],axis=1)

submission = submission.set_index('ForecastId')
submission.to_csv('submission.csv')

submission.to_csv('..\output\kaggle\working\submission_c.csv')
submission