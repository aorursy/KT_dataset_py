import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns; sns.set()

# Input data files are available in the "../input/" directory.





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#7日間の移動平均を予測するようにしたい。。。

train = pd.read_csv("../input/bitcoindata/bitcoin_price_Training - MA.csv")

test = pd.read_csv("../input/bitcoindata/bitcoin_price_Test - MA.csv")
train = train[::-1]

test = test[::-1]
train.head()
#x is input, y is teach data.

x_train = train['Close'].values.astype('float32')

y_train = train['Next7daysMA'].values.astype('float32')

x_test = test['Close'].values.astype('float32')

y_test = test['Next7daysMA'].values.astype('float32')
from sklearn.preprocessing import StandardScaler
# reshape data to scale the point

x_train = x_train.reshape(-1, 1)

y_train = y_train.reshape(-1, 1)

x_test = x_test.reshape(-1, 1)

y_test = y_test.reshape(-1, 1)
scaler = StandardScaler()

x_train_n = scaler.fit_transform(x_train)

y_train_n = scaler.fit_transform(y_train)

x_test_n = scaler.transform(x_test)

y_test_n = scaler.transform(y_test)
print(x_train_n.shape)

print(y_train_n.shape)

print(x_test_n.shape)

print(y_test_n.shape)
lookback = 24

step = 1

delay = 7

batch_size = 128

train_gen = generator(train_n, lookback=lookback, delay=delay,

    min_index=0, max_index=1000, shuffle=True, step=step,

batch_size=batch_size)

val_gen = generator(train_n, lookback=lookback, delay=delay,

    min_index=1001, max_index=None, step=step, batch_size=batch_size)

test_gen = generator(test_n, lookback=lookback, delay=delay,

    min_index=0, max_index=None, step=step, batch_size=batch_size)

# This is how many steps to draw from `val_gen` in order to see the whole validation set:

val_steps = (len(train_n) - 1001 - lookback) // batch_size

# This is how many steps to draw from `test_gen` in order to see the whole test set:

test_steps = (len(test_n) - lookback) // batch_size
# reproducibility (make sure each time training is occurred, the result will be the same)

np.random.seed(7)
from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop
model = Sequential()

model.add(layers.GRU(32,

                     dropout=0.2,

                     recurrent_dropout=0.2,

                     input_shape=(None, train_n.shape[-1])))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

model.summary()
history = model.fit_generator(train_gen,

                              steps_per_epoch=500,

                              epochs=40,

                              validation_data=val_gen,

                              validation_steps=val_steps)
loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'blue', label='train loss')

plt.plot(epochs, val_loss, 'orange', label='test loss')

plt.title('Training and validation loss')

plt.legend()
train_re = train_n.reshape(-1,1,1)

pred = model.predict(train_re)
pred = scaler.inverse_transform(pred)
plt.plot(range(len(train_re)), train, label='train')

plt.plot(range(len(train_re)), pred, label='prediction')

plt.legend()

plt.title("Prediction on training data")