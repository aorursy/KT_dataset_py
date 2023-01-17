%matplotlib inline



import os



import numpy as np

import seaborn as sns

import pandas as pd

from keras.preprocessing.image  import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from matplotlib import pyplot as plt





sns.set()
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.columns
ytrain = to_categorical(train.values[:, 0])

xtrain = train.values[:, 1:]

xtest = test.values



xtrain = xtrain.reshape((42000, 28, 28, 1)) / 255

xtest = xtest.reshape((28000, 28, 28, 1)) / 255



print('xtrain.shape:', xtrain.shape)

print('xtrain.max():', xtrain.max())

print('xtrain.min():', xtrain.min())

print('xtest.shape:', xtest.shape)
data_generator = ImageDataGenerator(rotation_range=10, zoom_range=0.10, width_shift_range=0.1, height_shift_range=0.1)

xtrain_news, ytrain_news = [], []



for _ in range(2):

    for i, sample in enumerate(xtrain):

        xtrain_new, ytrain_new = data_generator.flow(sample.reshape((1, 28, 28, 1)), ytrain[i].reshape((1, 10))).next()

        xtrain_news.append(xtrain_new.reshape((28, 28, 1)))

        ytrain_news.append(ytrain_new[0])

        

    

xtrain = np.concatenate([xtrain, np.array(xtrain_news)])

ytrain = np.concatenate([ytrain, np.array(ytrain_news)])

del xtrain_news, ytrain_news



xselftest = xtrain[:10000]

yselftest = ytrain[:10000]

xtrain = xtrain[10000:]

ytrain = ytrain[10000:]



print('Number of training samples:', len(xtrain))
model = Sequential()



model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=5, activation='relu', strides=2, padding='same'))

model.add(BatchNormalization())

model.add(Dropout(.4))



model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=5, activation='relu', strides=2, padding='same'))

model.add(BatchNormalization())

model.add(Dropout(.4))



model.add(Conv2D(128, kernel_size=4, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(.4))



model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
xtra, ytra = xtrain[1000:], ytrain[1000:]

xval, yval = xtrain[:1000], ytrain[:1000]

print('xtra.shape:', xtra.shape)

print('xval.shape:', xval.shape)
N_EPS = 20

h = model.fit(x=xtrain, y=ytrain, epochs=N_EPS, batch_size=64, validation_data=(xval, yval,), verbose=0)
plt.plot(range(N_EPS), h.history['loss'], marker='x', label='Loss');

plt.plot(range(N_EPS), h.history['val_loss'], color='green', label='Validation loss');

plt.title('Loss');

plt.legend();



print('Testing on', len(xselftest), 'samples')

print('Accuracy:', model.evaluate(xselftest, yselftest)[1] * 100)
ytestpred = model.predict(xtest).argmax(axis=1)



df = pd.read_csv('../input/sample_submission.csv')

df['Label'] = ytestpred

df.head()
df.to_csv('submission.csv', index=False)

pd.read_csv('submission.csv').head()