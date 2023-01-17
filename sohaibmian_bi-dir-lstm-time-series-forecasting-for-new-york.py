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
import tensorflow as tf

from tensorflow import keras

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
df.info()
df[df["Province_State"]=="New York"]
df1=df[df["Province_State"]=="New York"]
df1["days"]=[x for x in range(1,82)]
df1.tail()


time=df1.iloc[:,6]

y=df1.iloc[:,4]

time=time.to_numpy(dtype="float32")

series=y.to_numpy(dtype="float32")

time.shape
plt.figure(figsize=(20, 12))



plt.plot(time, series)

plt.title("Confirmed Cases in New York")

plt.ylabel("Confirmed Cases")

plt.xlabel("Days")
time=np.array(time)

series=np.array(series)

split_time = 78

time_train = time[:split_time]

x_train = series[:split_time]

time_valid = time[split_time:]

x_valid = series[split_time:]
window_size = 3

batch_size = 3

shuffle_buffer_size = 79
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    dataset = tf.data.Dataset.from_tensor_slices(series)

    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))

    dataset = dataset.batch(batch_size).prefetch(1)

  

    return dataset
dataset = windowed_dataset(x_train, window_size, batch_size=batch_size, shuffle_buffer=shuffle_buffer_size)
tf.keras.backend.clear_session()

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)



model = tf.keras.models.Sequential([

  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),

                      input_shape=[None]),

  tf.keras.layers.Bidirectional(tf.keras.layers.GRU(75, return_sequences=True,activation="relu")),

  tf.keras.layers.Bidirectional(tf.keras.layers.GRU(50,activation="relu")),

  tf.keras.layers.Dense(1),

  tf.keras.layers.Lambda(lambda x: x * 100.0)

])





model.compile(loss=tf.keras.losses.Huber(), optimizer="adam",metrics=["mae"])

history = model.fit(dataset,epochs=1000)
forecast=[]

for time in range(len(series) - window_size):

    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))



forecast = forecast[split_time-window_size:]

results = np.array(forecast)[:, 0, 0]

print(forecast)



plt.figure(figsize=(20, 12))



plt.plot( time_valid, x_valid,label="Real")

plt.plot(time_valid, results,label="Forecasted")

plt.title("Bidirectionl LSTM Forecasting")

plt.ylabel("Confirmed Cases")

plt.xlabel("Days")

plt.legend()
df2=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
df2.info()
df2=df2[df2["Province_State"]=="New York"]
df2.info()
df3=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
df3.head()
df2.head()
df4=pd.DataFrame()

df4["days_test"]=[x for x in range(1,124)]
df5=df1.reset_index()

df4=df4.reset_index()

df6 = [df5, df4]

df_test = pd.concat(df6, axis=1)
df_test.tail()
time=df_test.iloc[:,9]

y=df_test.iloc[:,5]

time=time.to_numpy(dtype="float32")

series=y.to_numpy(dtype="float32")
time=np.array(time)

series=np.array(series)

split_time = 79

time_valid = time[split_time:]

x_valid = series[split_time:]
window_size = 3

batch_size = 3
forecast=[]

for time in range(len(series) - window_size):

    z=model.predict(series[time:time + window_size][np.newaxis])

    if time >= 76:

        series[time+window_size]=z

    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    



forecast1 = forecast[split_time-window_size:]

results = np.array(forecast1)[:, 0, 0]





plt.figure(figsize=(20, 10))



plt.plot(time_valid, results,label="Forecasted")

plt.title("Bidirectionl GRU Forecasting")

plt.ylabel("Confirmed Cases")

plt.xlabel("Days")

plt.legend()
print(len(forecast1))
df3.info()
df2.info()
df_s=df3.merge(df2,on="ForecastId")
df_s.info()
forecast2=forecast1[1:]
forecast3=[int(i) for i in forecast2]
print(forecast3)
print(len(forecast3))
df_s["ConfirmedCases"]=forecast3
df_s.head()
df_s=df_s[["ForecastId","ConfirmedCases","Fatalities"]]
df_s.head()
df_s.tail()
L=0

for i in df_s["ForecastId"]:

    df3.iloc[i-1,1]=forecast3[L]

    L=L+1

df3.iloc[11610]
df3.to_csv("submission.csv",index=False)