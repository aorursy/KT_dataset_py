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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,RepeatVector
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
data['date'] = pd.to_datetime(data['date'])
data.info()
data.head()
data.set_index(['date'],inplace=True)
data = data['item_cnt_day'].resample('D').sum()
df=pd.DataFrame(data)
plt.figure(figsize=(16,8))
df['item_cnt_day'].plot(color = 'G')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.show()
df_1=df.values
df_1=df_1.astype('float32')

scaler = MinMaxScaler(feature_range=(-1,1))
ts = scaler.fit_transform(df_1)
df.info()
timestep = 30


X= []
Y=[]

raw_data=ts

for i in range(len(raw_data)- (timestep)):
    X.append(raw_data[i:i+timestep])
    Y.append(raw_data[i+timestep])


X=np.asanyarray(X)
Y=np.asanyarray(Y)


k = 850
Xtrain = X[:k,:,:]  
Ytrain = Y[:k]    
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(30, 1)))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='sgd', loss='mse')
# fit model
model.fit(Xtrain, Ytrain, epochs=200, verbose=0)

Xtest = X[k:,:,:]  
Ytest= Y[k:]
preds = model.predict(Xtest)
preds = scaler.inverse_transform(preds)


Ytest=np.asanyarray(Ytest)  
Ytest=Ytest.reshape(-1,1) 
Ytest = scaler.inverse_transform(Ytest)


Ytrain=np.asanyarray(Ytrain)  
Ytrain=Ytrain.reshape(-1,1) 
Ytrain = scaler.inverse_transform(Ytrain)

mean_squared_error(Ytest,preds)
from matplotlib import pyplot
pyplot.figure(figsize=(20,10))
pyplot.plot(Ytest, 'G')
pyplot.plot(preds, 'r')
pyplot.show()
print(preds)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0,
    reduction="auto",
    name="categorical_crossentropy",
)
cce = tf.keras.losses.CategoricalCrossentropy()
cce(Ytest, preds).numpy()
logit = model(Xtest)
loss_value = loss_fn(Ytest, logit)
loss_value
Y_test = tf.convert_to_tensor(Ytest)
# print(model(Xtest))
print(Ytest)

