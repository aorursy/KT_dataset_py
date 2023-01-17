# For file manipulation

import os



# For data manipulation

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



# For our CNN model

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D
# Loading the train and test data

x_train = np.load('../input/reducing-image-sizes-to-32x32/X_train.npy')

x_test = np.load('../input/reducing-image-sizes-to-32x32/X_test.npy')

y_train = np.load('../input/reducing-image-sizes-to-32x32/y_train.npy')
# Preprocessing the image data

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255.

x_test /= 255.
# Defining the required variables

batch_size = 64

num_classes = 14

epochs = 30

val_split = 0.1

input_shape=x_train.shape[1:]
def baseline_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))



    model.add(Conv2D(64, (3, 3), padding='same'))

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.25))



    model.add(Flatten())

    

    model.add(Dense(1024))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(1024))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    

    model.add(Dense(num_classes))

    model.add(Activation('softmax'))

    

    return model
model = baseline_model()
# Compiling the model

model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)
# Training the model

hist = model.fit(

    x_train, 

    y_train,

    batch_size=batch_size,

    epochs=epochs,

    validation_split=val_split,

    shuffle=True

)
history = hist.history



fig, ax = plt.subplots(2)



ax[0].plot(history['acc'])

ax[0].plot(history['val_acc'])

ax[0].legend(['training accuracy', 'validation accuracy'])



ax[1].plot(history['loss'])

ax[1].plot(history['val_loss'])

ax[1].legend(['training loss', 'validation loss'])



for axs in ax.flat:

    axs.label_outer()
y_test = model.predict(x_test)



submission_df = pd.read_csv('../input/iwildcam-2019-fgvc6/sample_submission.csv')

submission_df['Predicted'] = y_test.argmax(axis=1)

print(submission_df.shape)

submission_df.head()
submission_df.to_csv('submission.csv',index=False)

# history_df.to_csv('history.csv', index=False)



# with open('history.json', 'w') as f:

#     json.dump(hist.history, f)