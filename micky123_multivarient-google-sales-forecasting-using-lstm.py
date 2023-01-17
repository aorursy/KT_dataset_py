# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout , LSTM , Bidirectional 

import tensorflow.compat.v1 as tf
print(tf.test.gpu_device_name())
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from sklearn.preprocessing import MinMaxScaler
train = pd.read_csv("../input/google-stock-price/Google_Stock_Price_Train.csv")
test = pd.read_csv("../input/google-stock-price/Google_Stock_Price_Test.csv")
train.head()
test.head()
train = train.set_index("Date")
test = test.set_index("Date")
train["Volume"] = train["Volume"].replace(",", "",regex=True)
train["Close"] = train["Close"].replace(",", "",regex=True)

train["Volume"] = train["Volume"].astype("float")
train["Close"] = train["Close"].astype("float")
print("train dataset shape", train.shape)
print("test dataset shape", test.shape)
train.info()
test["Volume"] = test["Volume"].replace(",", "",regex=True)
test["Close"] = test["Close"].replace(",", "",regex=True)

test["Volume"] = test["Volume"].astype("float")
test["Close"] = test["Close"].astype("float")
test.info()
scale = MinMaxScaler()

num_col = ["High", "Low", "Close", "Volume"]
train1 = scale.fit(train[num_col].to_numpy())

train.loc[:, num_col] = train1.transform(train[num_col].to_numpy())
test.loc[:,num_col] = train1.transform(test[num_col].to_numpy())


#Output variable
scale1 = MinMaxScaler()
Open = scale1.fit(train[["Open"]])
train["Open"] = Open.transform(train[["Open"]].to_numpy())
test["Open"] = Open.transform(test[["Open"]].to_numpy())
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()
def prepare_data(X,y,time_steps=1):
    Xs = []
    Ys = []
    for i in tqdm(range(len(X) - time_steps)):
        a = X.iloc[i:(i + time_steps)].to_numpy()
        Xs.append(a)
        Ys.append(y.iloc[i+time_steps])
    return np.array(Xs),np.array(Ys)    
steps = 10
X_train , y_train = prepare_data(train,train.Open,time_steps=steps)
X_test , y_test = prepare_data(test,test.Open,time_steps=steps)
print("X_train : {}\nX_test : {}\ny_train : {}\ny_test: {}".format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
X_train = np.asarray(X_train).astype(np.float32)
model = Sequential()
model.add(LSTM(128,input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.2))

model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer="adam",loss="mse")

with tf.device('/GPU:0'):
    prepared_model = model.fit(X_train,y_train,batch_size=32,epochs=1000,validation_data=(X_test,y_test))

plt.plot(prepared_model.history["loss"],label="loss")
plt.plot(prepared_model.history["val_loss"],label="val_loss")
plt.legend(loc="best")
plt.xlabel("No. Of Epochs")
plt.ylabel("mse score")
plt.plot(prepared_model.history["loss"],label="loss")
plt.plot(prepared_model.history["val_loss"],label="val_loss")
plt.legend(loc="best")
plt.xlabel("No. Of Epochs")
plt.ylabel("mse score")
pred = model.predict(X_test)

y_test_inv = scale1.inverse_transform(y_test.reshape(-1,1))
pred_inv = scale1.inverse_transform(pred)

plt.figure(figsize=(16,6))
plt.plot(y_test_inv.flatten(),marker=".",label="actual")
plt.plot(pred_inv.flatten(),marker=".",label="prediction",color="r")
y_test_actual = scale1.inverse_transform(y_test.reshape(-1,1))
y_test_pred = scale1.inverse_transform(pred)

arr_1 = np.array(y_test_actual)
arr_2 = np.array(y_test_pred)

actual = pd.DataFrame(data=arr_1.flatten(),columns=["actual"])
predicted = pd.DataFrame(data=arr_2.flatten(),columns = ["predicted"])
final = pd.concat([actual,predicted],axis=1)
final.head()
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(final.actual,final.predicted)) 
r2 = r2_score(final.actual,final.predicted) 
print("rmse is : {}\nr2 is : {}".format(rmse,r2))
plt.figure(figsize=(16,6))
plt.plot(final.actual,label="Actual data")
plt.plot(final.predicted,label="predicted values")
plt.legend(loc="best")


