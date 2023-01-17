import pandas as pd

import keras

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Flatten

from keras.optimizers import SGD, Adam, RMSprop

from keras.utils import np_utils

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras import backend as K

import matplotlib.pyplot as plt
batch_size = 128 # number of samples per each training in the dataset

num_classes = 10 # number of Y

epochs = 2 # number of training; assign 30 or more for better accuracy
train = pd.read_csv('../input/train.csv').values

trainY = np_utils.to_categorical(train[:,0].astype('int32'), num_classes) # labels

trainX = train[:, 1:].astype('float32') # Pixel values

trainX /= 255 # Normalize values for training

# Visulize 2D converted values

viz_trainX = trainX.reshape(trainX.shape[0], 28, 28)

for image in viz_trainX[:5]:

    plt.imshow(image, cmap='gray')

    plt.show()
# turn data to 3D tensor with shape of (28, 28, 1)

img_rows, img_cols = 28, 28



trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)



input_shape = (img_rows, img_cols, 1)


model = Sequential()

model.add(Conv2D(32,

                 data_format='channels_last',

                 kernel_size=(3,3),

                 activation='relu',

                 input_shape=input_shape)

                )

model.add(Conv2D(64, (3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))





# Compile model with Optimizer

model.compile(

              loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy']

             )
from keras.callbacks import EarlyStopping

from keras.callbacks import TensorBoard

callbacks = [

                # EarlyStopping stops training when model does not improve after epochs. interval can be configured with patience.

                EarlyStopping(

                              monitor="loss",

                              min_delta=0,

                              patience=3, 

                              verbose=0,

                              mode='auto'

                             ),

                # Tensorboard saves Tensorboard of the model after training

                TensorBoard(

                            log_dir='./logs',

                            histogram_freq=0,

                            batch_size=batch_size,

                            write_graph=True,

                            write_grads=False,

                            write_images=False, 

                            embeddings_freq=0, 

                            embeddings_layer_names=None,

                            embeddings_metadata=None

                           )

            ]
model.fit(

          trainX, trainY,

          batch_size=batch_size,

          epochs=epochs,

          verbose=2,

          callbacks=callbacks

         )



score = model.evaluate(trainX, trainY, verbose=0)

print('Train loss:', score[0])

print('Train accuracy:', score[1])
# Change test data to Tensor with input shape

testX = pd.read_csv('../input/test.csv').values.astype('float32')

testX /= 255 # normalize each pixel

testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
# Predict with trained model and record results on csv file

testY = model.predict_classes(testX, verbose=2)



pd.DataFrame({"ImageId": list(range(1,len(testY)+1)),

              "Label": testY}

            ).to_csv('submission.csv', index=False, header=True)