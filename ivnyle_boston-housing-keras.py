import keras

import numpy as np

from sklearn.model_selection import train_test_split
data = np.load('../input/boston_housing.npz')

print(data['x'].shape)

print(data['y'].shape)
train_data, test_data, train_targets, test_targets = train_test_split(data['x'], data['y'], test_size=0.2, random_state=42)
print(train_data.shape)

print(test_data.shape)

print(train_targets.shape)

print(test_targets.shape)
mean = train_data.mean(axis=0)

train_data -= mean

std = train_data.std(axis=0)

train_data /= std



test_data -= mean

test_data /= std
from keras import models

from keras import layers



def build_model():

    model = models.Sequential()

    model.add(layers.Dense(64, activation='relu',

                          input_shape=(train_data.shape[1],)))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model
import time

start = time.time()



k = 4

num_val_samples = len(train_data) // k

num_epochs = 100

all_scores = []



for i in range(k):

    print('processing fold #', i)

    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]

    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    

    partial_train_data = np.concatenate(

        [train_data[:i * num_val_samples],

         train_data[(i + 1) * num_val_samples:]],

         axis=0)

    

    partial_train_targets = np.concatenate(

        [train_targets[:i * num_val_samples],

         train_targets[(i + 1) * num_val_samples:]],

         axis=0)

    

    model = build_model()

    model.fit(partial_train_data, partial_train_targets,

             epochs=num_epochs, batch_size=1, verbose=0)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

    all_scores.append(val_mae)

    

end = time.time()

print(end - start)