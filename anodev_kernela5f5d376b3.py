import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
from keras.regularizers import l2

# Get the data
train = pd.read_csv("../input/train.csv").values
test  = pd.read_csv("../input/test.csv").values

Train_X = train[:, 1:].reshape(train.shape[0], 28, 28, 1)
Train_X = Train_X.astype(float)
Train_X /= 255.0
Train_Y = kutils.to_categorical(train[:, 0])

Classes = Train_Y.shape[1]

Test_X = test.reshape(test.shape[0], 28, 28, 1)
Test_X = Test_X.astype(float)
Test_X /= 255.0

cnn = models.Sequential()

cnn.add(conv.Convolution2D(16, 5, 5,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))
cnn.add(conv.Convolution2D(32, 3, 3, activation="relu", border_mode='same'))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(conv.Convolution2D(64, 3, 3, activation="relu", border_mode='same'))
#cnn.add(conv.Convolution2D(64, 3, 3, activation="relu", border_mode='same'))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(core.Flatten())
cnn.add(core.Dropout(0.2))
cnn.add(core.Dense(256, activation="relu", W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
cnn.add(core.Dropout(0.1))
cnn.add(core.Dense(128, activation="relu", W_regularizer=l2(0.01), b_regularizer=l2(0.01)))
cnn.add(core.Dense(Classes, activation="softmax"))

cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

cnn.fit(Train_X, Train_Y, batch_size=30, nb_epoch=12, verbose=1)

prediction = cnn.predict_classes(Test_X)

np.savetxt('cnn-RecogDigits.csv', np.c_[range(1,len(prediction)+1),prediction], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
