import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras import backend as K

from keras.layers import Conv2D, MaxPool2D

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dropout, Dense, Flatten, BatchNormalization

from keras.losses import categorical_crossentropy

from keras.optimizers import Adadelta

from keras.utils import to_categorical

np.random.seed(0)
train_data_path = '../input/train.csv'

test_data_path = '../input/test.csv'
train = pd.read_csv(train_data_path)

test = pd.read_csv(test_data_path)
data = train.drop('label', axis=1)

label = train[['label']]
data.shape, label.shape
X_train, X_test, y_train, y_test = train_test_split(data, label, shuffle=True, test_size=0.1)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
X_train = X_train.as_matrix()

X_test = X_test.as_matrix()

y_train = y_train.as_matrix()

y_test = y_test.as_matrix()
X_train.shape, y_train.shape
X_test.shape, y_test.shape
if K.image_data_format() == 'channels_first':

    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

else:

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)
X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

y_train = y_train.astype('float32')

y_test = y_test.astype('float32')
X_train.shape, X_test.shape, y_train.shape, y_test.shape
model = Sequential()
model.add(Conv2D(filters=32, 

                 kernel_size=(5, 5), 

                 activation='relu', 

                 input_shape=X_train[0].shape))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(filters=32, 

                 kernel_size=(3, 3), 

                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=-1))

model.add(Conv2D(filters=64,

                 kernel_size=(3, 3), 

                 activation='relu'))

model.add(BatchNormalization(axis=-1))

model.add(Conv2D(filters=64,

                 kernel_size=(3, 3), 

                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))
model.compile(loss=categorical_crossentropy, 

              optimizer=Adadelta(), 

              metrics=['accuracy'])
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,

                         height_shift_range=0.08, zoom_range=0.08)



test_gen = ImageDataGenerator()
train_generator = gen.flow(X_train, y_train, batch_size=64)

test_generator = test_gen.flow(X_test, y_test, batch_size=64)
model.fit_generator(train_generator, steps_per_epoch=X_train.shape[0]//64, epochs=10, 

                    validation_data=test_generator, validation_steps=X_test.shape[0]//64)
loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
loss, accuracy
test = test.as_matrix()
if K.image_data_format() == 'channels_first':

    test = test.reshape(test.shape[0], 1, 28, 28)

else:

    test = test.reshape(test.shape[0], 28, 28, 1)
ans = model.predict_classes(test, batch_size=32, verbose=True)
ans
res = pd.DataFrame(columns=['ImageId', 'Label'])
res['ImageId'] = pd.Series(1 + np.arange(ans.shape[0]))
res['Label'] = pd.Series(ans)
res.head()
res.to_csv('submission_cnn_mnist.csv', index=False)
X_total_train = np.vstack((X_train, X_test))

y_total_train = np.vstack((y_train, y_test))
X_total_train.shape, y_total_train.shape
total_loss, total_accuracy = model.evaluate(X_total_train, y_total_train, verbose=True)
total_loss, total_accuracy