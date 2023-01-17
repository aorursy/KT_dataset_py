# pandas version should be >= 0.24.1

!pip install --upgrade pandas
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras import models, layers, optimizers

from keras.models import Sequential

from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPool2D, Dropout

from keras.optimizers import Adam

from keras import backend as K



print('Pandas version: ' + pd.__version__)

print(K.tensorflow_backend._get_available_gpus())
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
y = train['label'].to_numpy().reshape((train.shape[0], 1))

y = to_categorical(y, num_classes=10)



del train['label']

X = train.to_numpy()



pred = test.to_numpy().reshape((-1, 28, 28, 1)) / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)



X_train = X_train / 255.0

X_test = X_test / 255.0



X_train = X_train.reshape((-1, 28, 28, 1))

X_test = X_test.reshape((-1, 28, 28, 1))
model = Sequential()



model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(X_train.shape[1:]), activation='tanh'))

model.add(Conv2D(32, kernel_size=(5, 5), activation='tanh'))

model.add(Dropout(0.2))

# model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(64, kernel_size=(3, 3), activation='tanh'))

model.add(Conv2D(64, kernel_size=(3, 3), activation='tanh'))

model.add(Dropout(0.2))

# model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512, activation='tanh'))

model.add(Dense(128, activation='tanh'))

# model.add(Dense(64, activation='tanh'))

model.add(Dense(10, activation='softmax'))



adam = Adam(lr=0.001)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=50)
model.evaluate(X_test, y_test, verbose=0)
predictions = np.argmax(model.predict(pred), axis=-1)

predictions = predictions.reshape((predictions.shape[0], 1))

submission = pd.DataFrame({'ImageId':[i for i in range(1, predictions.shape[0] +1)], 'Label':predictions[:,0]})

submission.to_csv('submission.csv', index=False)