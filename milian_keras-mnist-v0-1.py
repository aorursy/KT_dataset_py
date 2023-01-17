import numpy as np

import pandas as pd



from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras import backend as K
# load_data

train_data = pd.read_csv('../input/train.csv')



labels = train_data.label.values.astype('int32')

train_data = train_data.drop('label', axis=1).as_matrix().astype('float32')



test_data = pd.read_csv('../input/test.csv').as_matrix().astype('float32')



# make label vectors

y_train = np_utils.to_categorical(labels) 
# normalize 0-1, max is 255, so...

X_train = train_data / np.max(train_data)

X_test = test_data / np.max(train_data)



# maybe norm by mean too?

# X_train = train_data / np.mean(X_train)

# X_test = test_data / np.mean(X_train)
img_size = (28, 28)



X_train = X_train.reshape(X_train.shape[0], img_size[0], img_size[1], 1)

X_test = X_test.reshape(X_test.shape[0], img_size[0], img_size[1], 1)

image_shape = (img_size[0], img_size[1], 1)


# number of convolutional filters to use

filters = 8

pool = (4, 4)

kernel = (3, 3)

dense = 128

dp = 0.2



model = Sequential()

model.add(Convolution2D(filters, 

                        kernel[0], 

                        kernel[1],

                        border_mode='same',

                        input_shape=image_shape))

model.add(Activation('relu'))

model.add(Convolution2D(filters, kernel[0], kernel[1]))

model.add(Activation('relu'))

model.add(Convolution2D(filters, kernel[0], kernel[1]))

model.add(Activation('relu'))

model.add(Convolution2D(filters, kernel[0], kernel[1]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool))

model.add(Dropout(dp))

model.add(Flatten())

model.add(Dense(dense))

model.add(Activation('relu'))

model.add(Dropout(dp))

model.add(Dense(y_train.shape[1]))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adadelta',

              metrics=['accuracy'])
print("Moneky see...")

model.fit(X_train, 

          y_train, 

          nb_epoch=10, 

          batch_size=32, 

          validation_split=0.1, 

          shuffle=True,

          verbose=1)
print("Monkey test...")

y_hat = model.predict_classes(X_train, verbose=0)

acc = sum(y_hat==labels)/len(labels)

best_so_far = 0.993119047619

print(acc, acc - best_so_far)
print("Monkey do...")

predictions = model.predict_classes(X_test, verbose=0)
save_me = pd.DataFrame({"Label": predictions})

save_me.index = save_me.index + 1



save_me.to_csv("keras_mnist_v0.1.csv",index_label='ImageId')



save_me.head()