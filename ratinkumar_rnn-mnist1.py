# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')
# Load the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# free some space

del train 



g = sns.countplot(Y_train)



Y_train.value_counts()
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0
x_train = []

for row in X_train.values:

#     x_train.append(

#         [

#             row[len(row)//2:],

#             row[:len(row)//2]

#         ]

#     )

    x_train.append(

        np.array(

            np.array(np.array_split(row, 4))

        )

    )

x_train = np.array(x_train)
x_train.shape, X_train.shape
timesteps, n_features = x_train.shape[1], x_train.shape[2]

x_train.shape, timesteps, n_features
from keras.layers import LSTM, RepeatVector, TimeDistributed

# lstm_autoencoder = Sequential()

# Encoder

# lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(2, 392), return_sequences=True))

# lstm_autoencoder.add(LSTM(124, activation='relu', return_sequences=False))

# lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=False))

# lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))

# lstm_autoencoder.add(RepeatVector(2))

# # Decoder

# lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))

# lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))

# lstm_autoencoder.add(LSTM(124, activation='relu', return_sequences=True))

# lstm_autoencoder.add(LSTM(392, activation='relu', return_sequences=False))



# lstm_autoencoder.summary()





# lstm_autoencoder = Sequential()

# # Encoder

# lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))

# lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))

# lstm_autoencoder.add(RepeatVector(timesteps))

# # Decoder

# lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))

# lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))

# lstm_autoencoder.add(TimeDistributed(Dense(n_features)))



# lstm_autoencoder.summary()



lstm_autoencoder = Sequential()

# Encoder

lstm_autoencoder.add(LSTM(256, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))

lstm_autoencoder.add(LSTM(128, activation='relu', return_sequences=True))

lstm_autoencoder.add(LSTM(64, activation='relu', return_sequences=False))

lstm_autoencoder.add(RepeatVector(timesteps))

# Decoder

lstm_autoencoder.add(LSTM(64, activation='relu', return_sequences=True))

lstm_autoencoder.add(LSTM(128, activation='relu', return_sequences=True))

lstm_autoencoder.add(LSTM(256, activation='relu', return_sequences=True))

lstm_autoencoder.add(TimeDistributed(Dense(n_features)))



lstm_autoencoder.summary()


from keras import optimizers

from keras.callbacks import ModelCheckpoint, TensorBoard



lr = 0.0001



adam = optimizers.Adam(lr)

lstm_autoencoder.compile(loss='mse', optimizer=adam)



cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",

                               save_best_only=True,

                               verbose=0)







tb = TensorBoard(log_dir='./logs',

                histogram_freq=0,

                write_graph=True,

                write_images=True)



lstm_autoencoder_history = lstm_autoencoder.fit(x_train, x_train, 

                batch_size = 256, epochs = 50, 

                shuffle = True, validation_split = 0.20);
a = lstm_autoencoder.predict(x_train[50].reshape(-1,4,196)).reshape(28, 28)

b = x_train[50].flatten().reshape(28, 28)
a = np.concatenate(a , axis=0).reshape(1,-1)

b = np.concatenate(b , axis=0).reshape(1,-1)
import matplotlib.pyplot as plt

plt.imshow(a)
values = [i for i in range(10)]

indexes = np.array([np.where(Y_train == number)[0] for number in values])

number_to_plot = 9

fig, ax = plt.subplots(nrows=5, ncols=2)

for i, row in enumerate(ax):

    for j, col in enumerate(row):

        col.imshow(x_train[indexes[number_to_plot][i]].reshape(28, 28))
fig, ax = plt.subplots(nrows=5, ncols=2)

for i, row in enumerate(ax):

    for j, col in enumerate(row):

        col.imshow(lstm_autoencoder.predict(x_train[indexes[number_to_plot][i]].reshape(-1,4,196)).reshape(28, 28))
# Encoded representation

encoded_rep = []

from keras.models import Model

encoder = Model(input=lstm_autoencoder.input , output=lstm_autoencoder.layers[2].output)
encoded_0 = encoder.predict(x_train[indexes[0]])
encoded = encoder.predict(x_train)
encoded_0.shape
encoded.shape
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=0).fit(encoded)
kmeans.predict(encoded_9)
plt.imshow(x_train[indexes[number_to_plot][i]].reshape(28,28))
x_train[0].shape

x_train[0][0].shape
encoded_9 = encoder.predict((x_train[indexes[number_to_plot][i]].reshape(-1,4,196)))