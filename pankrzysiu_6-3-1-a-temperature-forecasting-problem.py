# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import keras

keras.__version__
import os



data_dir = '../input'

fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')



f = open(fname)

data = f.read()

f.close()



lines = data.split('\n')

header = lines[0].split(',')

lines = lines[1:]



print(header)

print(len(lines))
import numpy as np



# Parse the data



float_data = np.zeros((len(lines), len(header) - 1))

for i, line in enumerate(lines):

    values = [float(x) for x in line.split(',')[1:]]

    float_data[i, :] = values

print(float_data.shape)
from matplotlib import pyplot as plt



# Temperature over the full temporal range of the dataset. Yearly trends are clearly visible



temp = float_data[:, 1] # <1> temperature (in degrees Celsius)

plt.plot(range(len(temp)), temp)

plt.show()
# The first 10 days of the temperature data



plt.plot(range(1440), temp[:1440])

plt.show()
# Normalizing the data



mean = float_data[:200000].mean(axis=0)

float_data -= mean

std = float_data[:200000].std(axis=0)

float_data /= std
# generator function used to feed the training, validation and test data



def generator(data, lookback, delay, min_index, max_index,

                shuffle=False, batch_size=128, step=6):

    if max_index is None:

        max_index = len(data) - delay - 1

    i = min_index + lookback

    while 1:

        if shuffle:

            rows = np.random.randint(

                    min_index + lookback, max_index, size=batch_size)

        else:

            if i + batch_size >= max_index:

                i = min_index + lookback

            rows = np.arange(i, min(i + batch_size, max_index))

            i += len(rows)



        samples = np.zeros((len(rows),

                            lookback // step,

                            data.shape[-1]))

        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):

            indices = range(rows[j] - lookback, rows[j], step)

            samples[j] = data[indices]

            targets[j] = data[rows[j] + delay][1]

        yield samples, targets
lookback = 1440

step = 6

delay = 144

batch_size = 128



train_gen = generator(float_data,

                        lookback=lookback,

                        delay=delay,

                        min_index=0,

                        max_index=200000,

                        shuffle=True,

                        step=step,

                        batch_size=batch_size)

val_gen = generator(float_data,

                        lookback=lookback,

                        delay=delay,

                        min_index=200001,

                        max_index=300000,

                        step=step,

                        batch_size=batch_size)

test_gen = generator(float_data,

                        lookback=lookback,

                        delay=delay,

                        min_index=300001,

                        max_index=None,

                        step=step,

                        batch_size=batch_size)



val_steps = (300000 - 200001 - lookback)  // batch_size # How many steps to draw from

            # val_gen in order to see the entire validation set

test_steps = (len(float_data) - 300001 - lookback)  // batch_size # How many steps to draw

        # from test_gen in order to see the entire test set 

from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop



model = Sequential()

model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(1))

# model.summary()



model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,

                            steps_per_epoch=500,

                            epochs=20,

                            validation_data=val_gen,

                            validation_steps=val_steps)

import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop



model = Sequential()

model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

model.summary()

history = model.fit_generator(train_gen,

                steps_per_epoch=500,

                epochs=20,

                validation_data=val_gen,

                validation_steps=val_steps)
import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop



model = Sequential()

model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))

model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

model.summary()

history = model.fit_generator(train_gen,

                                steps_per_epoch=500,

                                epochs=10, # 40,

                                validation_data=val_gen,

                                validation_steps=val_steps)
# print(history.history)

import matplotlib.pyplot as plt



loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
step = 3

lookback = 720

delay = 144



train_gen = generator(float_data,

                lookback=lookback,

                delay=delay,

                min_index=0,

                max_index=200000,

                shuffle=True,

                step=step)

val_gen = generator(float_data,

                lookback=lookback,

                delay=delay,

                min_index=200001,

                max_index=300000,

                step=step)

test_gen = generator(float_data,

                lookback=lookback,

                delay=delay,

                min_index=300001,

                max_index=None,

                step=step)



val_steps = (300000 - 200001 - lookback) // batch_size # 128

test_steps = (len(float_data) - 300001 - lookback) // batch_size # 128
from keras.models import Sequential

from keras import layers

from keras.optimizers import RMSprop



model = Sequential()

model.add(layers.Conv1D(32, 5, activation='relu',

        input_shape=(None, float_data.shape[-1])))

model.add(layers.MaxPooling1D(3))

model.add(layers.Conv1D(32, 5, activation='relu'))

model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))

model.add(layers.Dense(1))

model.summary()



model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,

        steps_per_epoch=500,

        epochs=20,

        validation_data=val_gen,

        validation_steps=val_steps)
import matplotlib.pyplot as plt



for m in ['loss']:

    loss = history.history[m]

    val_loss = history.history['val_' + m]

    epochs = range(1, len(loss) + 1)

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training ' + m)

    plt.plot(epochs, val_loss, 'b', label='Validation ' + m)

    plt.title('Training and validation ' + m)

    plt.legend()



plt.show()
# Doesn't seem to work on Kaggle



%%javascript

Jupyter.notebook.session.delete();