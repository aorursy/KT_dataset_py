# If true, use sin curve as a dataset
Test = False
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
# Import the dataset and encode the date
df = pd.read_csv(os.path.join(dirname, filenames[1]))
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()
# split data into train and test set
train_size = 200
# Sin wave test
if Test:
    Real_Price = np.sin(np.linspace(0,500,len(Real_Price)))

df_train= Real_Price[:len(Real_Price)-train_size]
df_test= Real_Price[len(Real_Price)-train_size:]
print("train:{} test:{}".format(df_train.shape, df_test.shape))

# Data preprocess
#test
if not Test:
    training_set = df_train.values
else:
    training_set = df_train

# Rescale the data
# Large value in the input slow downs training
from sklearn.preprocessing import StandardScaler

training_set = np.reshape(training_set, (len(training_set), 1))
scaler = StandardScaler()
training_set = scaler.fit_transform(training_set)

X_train = training_set[:,0]
X , Y = list(), list()

# number of Timestep
Tx = 100
# number of batch
batchnum = 30000
# number of feature
n_features = 1

for batch in range(batchnum):
    idx = np.random.randint(0, X_train.shape[0] - Tx -1)
    X.append(X_train[idx:idx+Tx])
    Y.append(X_train[idx+Tx])
    
X = np.array(X)
X = X.reshape((X.shape[0], X.shape[1], n_features))
Y = np.array(Y)

print("X_train.shape:" , X.shape)
print("Y_train.shape:" , Y.shape)
## building model
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers import Conv1D, Dense, Activation, Dropout, Input, LSTM, GRU, Reshape, Lambda, RepeatVector,TimeDistributed, BatchNormalization
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K



model = Sequential()
model.add(Conv1D(filters=196,kernel_size=15,strides=4,input_shape = (None, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(rate=0.8))

model.add(GRU(units = 128,  return_sequences=True, activation="relu"))
model.add(Dropout(rate=0.8))
model.add(BatchNormalization())

model.add(GRU(units = 128, activation="relu"))
model.add(Dropout(rate=0.8))
model.add(BatchNormalization())

# model.add(Dense(1))
model.add(Dense(1))

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='mean_squared_error')

model.summary()

test = np.ones((1,30,1))
test_predict = model.predict(test)

print("test.shape:", test.shape, "\npredict.shape:", test_predict.shape)
# train
history = model.fit(X, Y, epochs=30)
# show loss curve
import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
plt.plot(history.history['loss'])
plt.show()
# show all training data
plt.plot(X_train[:])
plt.ylabel('some numbers')
plt.show()
# Test Data preprocess
if not Test:
    test_set = df_test.values
else:
    test_set = df_test
test_set = np.reshape(test_set, (len(test_set), 1))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
test_set = scaler.fit_transform(test_set)
test_set = test_set[:,0]



ini = test_set[0:100]
res = np.array([])
res= np.append(res, ini)
print(res)
for i in range(100):
    inp = np.reshape(ini, (1,len(ini),1))
    pred = model.predict(inp)
    ini = np.append(ini[1:], pred[-1])
    res = np.append(res, pred[-1])


res = np.reshape(res, (len(res), 1))

res = scaler.inverse_transform(res)
test_set = scaler.inverse_transform(test_set)
plt.figure(figsize=(20,10))
ax = plt.gca()  

plt.plot(res[:],label = 'Predicted BTC Price')
plt.plot(test_set[:], label = 'Real BTC Price')

# test_set = test_set.reset_index()
# x=test_set.index
# labels = test_set['date']
# plt.xticks(x, labels, rotation = 'vertical')
# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(18)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontsize(18)
    
plt.legend(loc=2, prop={'size': 25})
plt.show()


