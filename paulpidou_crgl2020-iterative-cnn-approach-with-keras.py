import numpy as np

import pandas as pd

import os

from matplotlib import pyplot as plt



from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras import activations

from keras.utils import Sequence
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_val = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/train.csv', index_col='id')

train_val['delta'].describe()
MAX_DELTA = max(train_val['delta'])

GRID_HEIGHT = 25

GRID_WIDTH = 25
def line2grid(data):

    return data.to_numpy().reshape((data.shape[0], 1, GRID_HEIGHT, GRID_WIDTH))



start = line2grid(train_val.iloc[:,1:626])

stop = line2grid(train_val.iloc[:,626:])
# Taken from http://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/

def life_step(X):

    """Game of life step using generator expressions"""

    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)

                     for i in (-1, 0, 1) for j in (-1, 0, 1)

                     if (i != 0 or j != 0))

    return (nbrs_count == 3) | (X & (nbrs_count == 2))
idx = 25



# Start, Stop

fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle(f'Delta: {train_val.loc[idx, "delta"]}')

ax1.imshow(1-(start[idx, 0, :]), cmap="gray")

ax1.set_title("Start Setting")

ax2.imshow(1-(stop[idx, 0, :]), cmap="gray")

ax2.set_title("Stop Setting")



X = start[idx, 0]

delta = train_val.loc[idx, "delta"]



# Evolution over time (get missing steps)

fig, ax = plt.subplots(1, delta+1, figsize=((delta+1)*3 ,4))

fig.suptitle(f'Evolving over {delta} steps')

ax[0].imshow(1-X, cmap="gray")

ax[0].set_title("Start")



for i in range(delta):

    X = life_step(X)

    ax[i+1].imshow(1-X, cmap="gray")

    ax[i+1].set_title(f'Delta {i+1}')
class ImageSequence(Sequence):

    def __init__(self, start, delta, batch_size):

        self.start, self.delta = start, delta

        self.batch_size = batch_size



    def __len__(self):

        return int(np.ceil(len(self.start) * self.delta / float(self.batch_size))) - 1



    def __getitem__(self, idx):

        index = idx * self.batch_size // self.delta

        delta = idx * self.batch_size % self.delta

        #print(idx, start[index:].shape, delta)

        for curr_i, X in enumerate(self.start[index:, 0]):

            for i in range(delta, self.delta):

                if curr_i == 0 and i == delta:

                    batch_X = np.expand_dims(life_step(X), axis=0)

                    batch_Y = np.expand_dims(X, axis=0)

                else:

                    batch_X = np.append(batch_X, np.expand_dims(life_step(X), axis=0), axis=0)

                    batch_Y = np.append(batch_Y, np.expand_dims(X, axis=0), axis=0)

                X = life_step(X)

            

                if len(batch_X) == self.batch_size:

                    return np.expand_dims(batch_X, axis=-1), np.expand_dims(batch_Y, axis=-1)

            delta = 0
generator = ImageSequence(start, delta=5, batch_size=64).__iter__()

batch_X, batch_Y = next(generator)
size = 10



fig, ax = plt.subplots(1, size, figsize=((size)*3 ,4))

fig.suptitle(f'Batch X extract')



for i in range(size):

    ax[i].imshow(1-batch_X[i, :, :, 0], cmap="gray")

    ax[i].set_title(f'{i}')

    

fig, ax = plt.subplots(1, size, figsize=((size)*3 ,4))

fig.suptitle(f'Batch Y extract')



for i in range(size):

    ax[i].imshow(1-batch_Y[i, :, :, 0], cmap="gray")

    ax[i].set_title(f'{i}')
model = keras.Sequential(

    [

        keras.Input(shape=(GRID_HEIGHT, GRID_WIDTH, 1)),

        layers.Conv2D(64, kernel_size=(3, 3), padding="SAME"),

        layers.BatchNormalization(),

        layers.Activation(activations.relu),

        layers.Conv2D(64, kernel_size=(3, 3), padding="SAME"),

        layers.BatchNormalization(),

        layers.Activation(activations.relu),

        layers.Conv2D(128, kernel_size=(5, 5), padding="SAME"),

        layers.BatchNormalization(),

        layers.Activation(activations.relu),

        layers.Conv2D(64, kernel_size=(3, 3), padding="SAME"),

        layers.BatchNormalization(),

        layers.Activation(activations.relu),

        layers.Conv2D(64, kernel_size=(3, 3), padding="SAME"),

        layers.BatchNormalization(),

        layers.Activation(activations.relu),

        layers.Conv2D(1, kernel_size=(3, 3), padding="SAME"),

        layers.Activation(activations.sigmoid)

    ]

)



model.summary()
sequence = ImageSequence(start, delta=20, batch_size=512)
epochs = 1



model.compile(loss="bce", optimizer="rmsprop", metrics=["accuracy"])

history = model.fit_generator(sequence, epochs=epochs)
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train'], loc='upper left')

plt.show()
def get_predicted_start(stop, delta):

    predicted = model.predict(stop.reshape((1, GRID_HEIGHT, GRID_WIDTH, 1)))

    for i in range(delta-1):

        predicted = model.predict(predicted)

    return predicted
idx = 10



predicted = get_predicted_start(stop[idx], train_val.loc[idx, "delta"])



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9 ,4))

fig.suptitle(f'Delta: {train_val.loc[idx, "delta"]}')

ax1.imshow(1-(stop[idx, 0, :]), cmap="gray")

ax1.set_title("Stop Setting")

ax2.imshow(1-(start[idx, 0, :]), cmap="gray")

ax2.set_title("Start Setting")

ax3.imshow(1-(predicted[0, :, :, 0]>=0.5), cmap="gray")

ax3.set_title("Predicted Setting")
test_val = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/test.csv', index_col='id')
max_delta = max(test_val['delta'])

buckets = {}



for d in range(1, max_delta+1):

    buckets[d] = line2grid(test_val.loc[test_val['delta'] == d].iloc[:,1:626])

    print(d, len(buckets[d]))
def make_predictions(stop, delta):

    predictions = model.predict(stop.reshape((-1, GRID_HEIGHT, GRID_WIDTH, 1)))

    for i in range(delta-1):

        predictions = model.predict(predictions)

    return predictions
def prepare_submission(buckets, threshold=0.5, sample_submission_path='/kaggle/input/conways-reverse-game-of-life-2020/sample_submission.csv'):

    submission_file = pd.read_csv(sample_submission_path, index_col='id', nrows=1)

    dfs = []

    for d, imgs in buckets.items():

        print('Making prediction for bucket', d)

        predictions = make_predictions(imgs, d)

        predictions = (predictions > threshold).astype(int).reshape(-1, 25*25)

        df = pd.DataFrame(predictions, index=test_val.loc[test_val['delta'] == d].index.tolist(), columns=submission_file.columns.tolist())

        dfs.append(df)

    return pd.concat(dfs).sort_index()
submission = prepare_submission(buckets)
submission.to_csv('submission.csv', index_label='id')

submission