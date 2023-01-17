import numpy as np

import pandas as pd

import scipy.special

import matplotlib.pyplot as plt

import os

print(os.listdir("../input/Kannada-MNIST"))
from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
path_in = '../input/Kannada-MNIST/'
train_data = pd.read_csv(path_in+'train.csv')

val_data = pd.read_csv(path_in+'Dig-MNIST.csv')

test_data = pd.read_csv(path_in+'test.csv')

samp_subm = pd.read_csv(path_in+'sample_submission.csv')
dict_data = dict(zip(range(0, 10), (((train_data['label'].value_counts()).sort_index())).tolist()))

names = list(dict_data.keys())

values = list(dict_data.values())

plt.bar(names, values)

plt.grid()

plt.show()
dict_data = dict(zip(range(0, 10), (((val_data['label'].value_counts()).sort_index())).tolist()))

names = list(dict_data.keys())

values = list(dict_data.values())

plt.bar(names, values)

plt.grid()

plt.show()
print('# train samples:', len(train_data.index))

print('# val samples:', len(val_data.index))

print('# test samples:', len(test_data.index))
X_train = train_data.copy()

y_train = train_data['label']

del X_train['label']

X_val = val_data.copy()

y_val = val_data['label']

del X_val['label']

X_test = test_data.copy()

del X_test['id']

y_train = to_categorical(y_train, num_classes = 10)

y_val = to_categorical(y_val, num_classes = 10)
X_train = X_train.values.reshape(-1,28,28,1)

X_val = X_val.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
X_train = X_train.astype('float32')/255

X_val = X_val.astype('float32')/255

X_test = X_test.astype('float32')/255
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2019)
plt.imshow(X_train[5][:,:,0], cmap='gray')

plt.show()
plt.imshow(X_val[5][:,:,0], cmap='gray')

plt.show()
plt.imshow(X_test[5][:,:,0], cmap='gray')

plt.show()
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=2))



model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=2, padding='same'))



model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))

model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))

model.add(MaxPool2D(pool_size=2, padding='same'))



model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dense(10, activation='softmax'))
optimizer = RMSprop(lr=0.001)


model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
model.summary()
epochs = 50

batch_size = 512
datagen = ImageDataGenerator(

        featurewise_center=False,

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False,

        rotation_range=10,

        zoom_range = 0.10,

        width_shift_range=0.15,

        height_shift_range=0.15,

        horizontal_flip=False,

        vertical_flip=False)

datagen.fit(X_train)
# Fit the model

history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),

                    epochs=epochs,

                    validation_data=(X_val,y_val),

                    steps_per_epoch=X_train.shape[0] // batch_size)
y_test = model.predict(X_test)
y_test_classes = np.argmax(y_test, axis = 1)
output = pd.DataFrame({'id': samp_subm['id'],

                       'label': y_test_classes})

output.to_csv('submission.csv', index=False)
loss = history.history['loss']

loss_val = history.history['val_loss']

epochs = range(1, len(loss)+1)

plt.plot(epochs, loss, 'bo', label='loss_train')

plt.plot(epochs, loss_val, 'b', label='loss_val')

plt.title('value of the loss function')

plt.xlabel('epochs')

plt.ylabel('value of the loss function')

plt.legend()

plt.grid()

plt.show()
acc = history.history['acc']

acc_val = history.history['val_acc']

epochs = range(1, len(loss)+1)

plt.plot(epochs, acc, 'bo', label='accuracy_train')

plt.plot(epochs, acc_val, 'b', label='accuracy_val')

plt.title('accuracy')

plt.xlabel('epochs')

plt.ylabel('value of accuracy')

plt.legend()

plt.grid()

plt.show()
del model