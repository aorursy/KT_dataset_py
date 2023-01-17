import pandas as pd

import numpy as np



def create_submission(test_preds, file_name = "submission.csv"):

    submission = pd.concat([pd.Series(np.arange(1,len(test_preds) + 1)),pd.Series(test_preds)], axis = 1)

    submission.columns = ['ImageId','Label']

    submission.to_csv(file_name, index = False)
import pandas as pd

import numpy as np

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
import keras

from keras.regularizers import l2 # Perhaps not use?

from keras.models import Model, Sequential

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers import Dropout, Dense, Activation, Embedding, Input, Reshape, Flatten, UpSampling2D,AveragePooling2D,Layer

from keras.utils import to_categorical





x_train = np.array(train.drop(["label"], axis = 1))

y_train = np.array(train['label'])

x_test = np.array(test)

# Convert into a nice input shape for the neural net. 

x_train = x_train.reshape(42000,28,28,1)

x_test = x_test.reshape(28000,28,28,1)



# Convert the training labels to categorical. 

y_train = to_categorical(y_train)
# This code is completely copied from Yassine Ghouzam's kernel - thank you Yassine. 

# His kernel can be found here: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
model = Sequential()

model.add(Conv2D(filters=4, kernel_size = (5,5), strides = 1, padding = "same",  input_shape=(28,28,1), activation ="tanh"))

model.add(Conv2D(filters=8, kernel_size = (4,4), strides = 2, padding = "same", activation = "relu"))

model.add(Conv2D(filters=12, kernel_size = (4,4), strides = 2, padding = "same", activation = "relu"))

model.add(Flatten())

model.add(Dense(units = 200))

model.add(Dropout(0.5))

model.add(Dense(units = 10, activation = "softmax"))

model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

model.summary()
n_epochs = 10

model.fit_generator(datagen.flow(x_train,y_train,batch_size = 80), epochs = n_epochs,steps_per_epoch=x_train.shape[0])
y_preds = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

import numpy as np

def get_preds(preds):

    return np.apply_along_axis(np.argmax, 1, preds)





create_submission(get_preds(y_preds), file_name = "submission_regular_CNN.csv")
# This implementation is taken from StackOverflow and was posted by today@StackOverflow. All kudos to him. 

# It can be found at https://stackoverflow.com/questions/53855941/how-to-implement-rbf-activation-function-in-keras

from keras.layers import Layer

from keras import backend as K



class RBFLayer(Layer):

    def __init__(self, units, gamma, **kwargs):

        super(RBFLayer, self).__init__(**kwargs)

        self.units = units

        self.gamma = K.cast_to_floatx(gamma)



    def build(self, input_shape):

        self.mu = self.add_weight(name='mu',

                                  shape=(int(input_shape[1]), self.units),

                                  initializer='uniform',

                                  trainable=True)

        super(RBFLayer, self).build(input_shape)



    def call(self, inputs):

        diff = K.expand_dims(inputs) - self.mu

        l2 = K.sum(K.pow(diff,2), axis=1)

        res = K.exp(-1 * self.gamma * l2)

        return res



    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.units)

le_net5 = Sequential()

le_net5.add(Conv2D(filters = 6, kernel_size = (5,5), strides = 1,activation = "tanh", input_shape=(28,28,1), padding = "same"))

le_net5.add(AveragePooling2D(pool_size=(2,2), strides = 2, padding = "valid"))

le_net5.add(Conv2D(filters = 16, kernel_size = (5,5), strides = 1, activation = "tanh"))

le_net5.add(Dropout(0.05))

le_net5.add(AveragePooling2D(pool_size=(2,2), strides = 2, padding = "same"))

le_net5.add(Conv2D(filters = 120, kernel_size = (5,5), strides = 1, activation = "tanh"))

le_net5.add(Dropout(0.05))

le_net5.add(Flatten())

le_net5.add(Dense(units = 84, activation = "tanh"))

le_net5.add(RBFLayer(10, gamma=0.5))

le_net5.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

le_net5.summary()
le_net5.fit_generator(datagen.flow(x_train, y_train), epochs = n_epochs,steps_per_epoch=x_train.shape[0])
y_preds = le_net5.predict(x_test)



create_submission(get_preds(y_preds), "submission_leNet55.csv")