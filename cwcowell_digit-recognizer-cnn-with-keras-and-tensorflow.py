from keras import callbacks

from keras import layers

from keras import models

from keras import regularizers

from keras.utils import to_categorical

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
%matplotlib inline
train_data_df = pd.read_csv('../input/train.csv')

train_data = train_data_df.values
X_train = train_data[:, 1:]
X_train = X_train.astype('float16') / 255.0
X_train = X_train.reshape((42000, 28, 28, 1))
y_train = train_data[:, 0]

y_train = to_categorical(y_train)
test_data_df = pd.read_csv('../input/test.csv')

test_data = test_data_df.values



X_test = test_data

X_test = X_test.astype('float16') / 255.0

X_test = X_test.reshape((28000, 28, 28, 1))
# use this callback during exploratory training

# stop_early_callback = callbacks.EarlyStopping(monitor='val_acc', 

#                                               patience=3, 

#                                               restore_best_weights=True,

#                                               verbose=1)



# use this callback during "production" training, when we don't set aside a validation dataset

stop_early_callback = callbacks.EarlyStopping(monitor='acc', 

                                              patience=5, 

                                              restore_best_weights=True,

                                              verbose=1)



callbacks_list = [stop_early_callback]
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu')) 

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))



model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dropout(0.4))

model.add(layers.Dense(10, activation='softmax'))



model.compile(optimizer='rmsprop', 

              loss='categorical_crossentropy', 

              metrics=['acc'])



validation_split = 0.0 # we'll need this variable later, when deciding whether to graph our validation figures



history = model.fit(X_train, 

                    y_train, 

                    callbacks=callbacks_list, 

                    epochs=30, 

                    batch_size=256, 

                    validation_split=validation_split, 

                    verbose=2)
epochs = range(1, len(history.history['loss']) + 1)

plt.plot(epochs, history.history['loss'], 'ro', label='training loss')

if validation_split > 0:

    plt.plot(epochs, history.history['val_loss'], 'r', label='val loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.show()



plt.plot(epochs, history.history['acc'], 'bo', label='training acc')

if validation_split > 0:

    plt.plot(epochs, history.history['val_acc'], 'b', label='val acc')

plt.xlabel('epochs')

plt.ylabel('acc')

plt.legend()
predictions_one_hot = model.predict(X_test)
predictions_ints = [np.argmax(prediction_one_hot) for prediction_one_hot in predictions_one_hot]
with open('predictions.csv', 'w') as predictions_file:

    predictions_file.write('ImageId,Label' + '\n')

    for i in range(len(test_data)):

        predictions_file.write(f'{i + 1},{predictions_ints[i]}' + '\n')