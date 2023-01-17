import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns; sns.set()

# Input data files are available in the "../input/" directory.





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/bitcoin_price_Training - Training.csv")

test = pd.read_csv("../input/bitcoin_price_1week_Test - Test.csv")



train = train[::-1]

test = test[::-1]



train = train['Close'].values.astype('float32')

test = test['Close'].values.astype('float32')
from sklearn.preprocessing import StandardScaler



# reshape data to scale the point

train = train.reshape(-1, 1)

test = test.reshape(-1, 1)



scaler = StandardScaler()

train_n = scaler.fit_transform(train)

test_n = scaler.transform(test)
def generator(data, lookback, delay, min_index, max_index, 

              shuffle=False, batch_size=128, step=1):

    if max_index is None:

        max_index = len(data) - delay - 1

    i = min_index + lookback

    while 1:

        if shuffle:

            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)

        else:

            if i + batch_size >= max_index:

                i = min_index + lookback

                

            rows = np.arange(i, min(i + batch_size, max_index))

            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))

        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):

            indices = range(rows[j] - lookback, rows[j], step)

            samples[j] = data[indices]

            targets[j] = data[rows[j] + delay]

        yield samples, targets
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
from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop
model = Sequential()

model.add(layers.GRU(32,

                     dropout=0.5,

                     recurrent_dropout=0.5,

                     return_sequences=True,

                     input_shape=(None, train_n.shape[-1])))

model.add(layers.GRU(64, activation='relu',

                     dropout=0.5,

                     recurrent_dropout=0.5))

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

plt.plot(epochs, val_loss, 'orange', label='validation loss')

plt.title('Training and validation loss')

plt.legend()
train_re = train_n.reshape(-1,1,1)

pred = model.predict(train_re)

pred = scaler.inverse_transform(pred)



plt.plot(range(len(train_re)), train, label='train')

plt.plot(range(len(train_re)), pred, label='prediction')

plt.legend()

plt.title("Prediction on training data")
