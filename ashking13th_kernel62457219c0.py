import numpy as np 

import pandas as pd

import os

from tqdm import tqdm



# Fix seeds

from numpy.random import seed

seed(639)

from tensorflow import set_random_seed

set_random_seed(5944)

float_data = pd.read_csv("../input/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values



# Helper function for the data generator. Extracts mean, standard deviation, and quantiles per time step.

# Can easily be extended. Expects a two dimensional array.

def extract_features(z):

     return np.c_[z.mean(axis=1), 

                  z.min(axis=1),

                  z.max(axis=1),

                  z.std(axis=1)]
def create_X(x, last_index=None, n_steps=150, step_length=1000):

    if last_index == None:

        last_index=len(x)

       

    assert last_index - n_steps * step_length >= 0



    # Reshaping and approximate standardization with mean 5 and std 3.

    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3

    

    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 

    # of the last 10 observations. 

    return np.c_[extract_features(temp),

                 extract_features(temp[:, -step_length // 10:]),

                 extract_features(temp[:, -step_length // 100:])]



# Query "create_X" to figure out the number of features

n_features = create_X(float_data[0:150000]).shape[1]

print("Our RNN is based on %i features"% n_features)
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):

    if max_index is None:

        max_index = len(data) - 1

     

    while True:

        # Pick indices of ending positions

        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)

         

        # Initialize feature matrices and targets

        samples = np.zeros((batch_size, n_steps, n_features))

        targets = np.zeros(batch_size, )

        

        for j, row in enumerate(rows):

            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

            targets[j] = data[row - 1, 1]

        yield samples, targets
batch_size = 32



# Position of second (of 16) earthquake. Used to have a clean split

# between train and validation

second_earthquake = 50085877

float_data[second_earthquake, 1]



# Initialize generators

train_gen = generator(float_data, batch_size=batch_size) # Use this for better score

# train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake + 1)

valid_gen = generator(float_data, batch_size=batch_size, max_index=second_earthquake)



# Define model

from keras.models import Sequential

from keras.layers import Dense, CuDNNGRU

from keras.optimizers import adam

from keras.callbacks import ModelCheckpoint



cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3)]



model = Sequential()

model.add(CuDNNGRU(48, input_shape=(None, n_features)))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))



model.compile(optimizer=adam(lr=0.0005), loss="mae")

model.summary()



from keras.utils import plot_model

plot_model(model)
history = model.fit_generator(train_gen,

                              steps_per_epoch=1000,

                              epochs=30,

                              verbose=0,

                              callbacks=cb,

                              validation_data=valid_gen,

                              validation_steps=200)
import matplotlib.pyplot as plt



def perf_plot(history, what = 'loss'):

    x = history.history[what]

    val_x = history.history['val_' + what]

    epochs = np.asarray(history.epoch) + 1

    

    plt.plot(epochs, x, 'bo', label = "Training " + what)

    plt.plot(epochs, val_x, 'b', label = "Validation " + what)

    plt.title("Training and validation " + what)

    plt.xlabel("Epochs")

    plt.legend()

    plt.show()

    return None



perf_plot(history)



# Load submission file

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})



# Load each test data, create the feature matrix, get numeric prediction

for i, seg_id in enumerate(tqdm(submission.index)):

  #  print(i)

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    x = seg['acoustic_data'].values

    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))



submission.head()



# Save

submission.to_csv('submission.csv')