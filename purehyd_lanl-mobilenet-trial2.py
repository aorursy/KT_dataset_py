# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import scipy.stats as stt

import scipy.signal as sig

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from numpy.random import seed

seed(639)

from tensorflow import set_random_seed

set_random_seed(5944)



from tqdm import tqdm

%%time

#import

float_data = pd.read_csv("../input/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values
float_data.shape
import matplotlib.pyplot as plt

%matplotlib inline
# For a given ending position "last_index", we split the last 150'000 values 

# of "x" into 150 pieces of length 1000 each. So n_steps * step_length should equal 150'000.

# From each piece, a set features are extracted. This results in a feature matrix 

# of dimension (150 time steps x features).  

def create_X(x, last_index=None, n_steps=150, step_length=1000):

    if last_index == None:

        last_index=len(x)

       

    assert last_index - n_steps * step_length >= 0

    

    # Reshaping and approximate standardization with mean 5 and std 3.

    temp = x[(last_index - n_steps * step_length):last_index]

    if temp.shape == (150000,2):

        temp = temp[:,0]

    

    f, t, temp_stft = sig.stft(temp, nperseg=1000)

    temp_stacked = np.stack((temp_stft.real, temp_stft.imag, np.abs(temp_stft)), axis=-1)

    return temp_stacked
# Query "create_X" to figure out the number of features

n_features = create_X(float_data[0:150000]).shape

print("Our RNN is based on {} features".format(n_features))
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):

    if max_index is None:

        max_index = len(data) - 1

     

    while True:

        # Pick indices of ending positions

        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)

         

        # Initialize feature matrices and targets

        samples = np.zeros((batch_size, *list(n_features)))

        targets = np.zeros(batch_size, )

        

        for j, row in enumerate(rows):

            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

            targets[j] = data[row - 1, 1]

        yield samples, targets

        

batch_size = 32
tot_len = float_data.shape[0]
train_d, test_d = float_data[:4 * tot_len//5], float_data[4 * tot_len//5:]
#second_earthquake = 50085877

#float_data[second_earthquake, 1]



# Initialize generators

train_gen = generator(train_d, batch_size=batch_size) # Use this for better score

valid_gen = generator(test_d, batch_size=batch_size)
from keras.models import Sequential

from keras.layers import Dense, CuDNNGRU

from keras.applications import vgg16, inception_v3, resnet50, mobilenet, xception

from keras.optimizers import adam

from keras.callbacks import ModelCheckpoint, EarlyStopping



cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3), 

      EarlyStopping(monitor='val_loss', patience=10, verbose=1)]



model = Sequential()

model.add(mobilenet.MobileNet(include_top=True, weights=None, input_shape=n_features))

model.add(Dense(20, activation='relu'))

model.add(Dense(1))



model.summary()
model.compile(optimizer=adam(lr=0.0005), loss="mae")
history = model.fit_generator(train_gen,

                              steps_per_epoch=100,

                              epochs=200,

                              verbose=1,

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