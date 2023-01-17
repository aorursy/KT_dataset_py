# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization 
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import get_custom_objects

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
%matplotlib inline

# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

x_train = train.iloc[:,1:].values.astype('float64')
y_train = train.iloc[:,0].values.astype('int32')
test = test.values.astype('float64')

m = x_train.shape[0]

# Normalize and reshape
x_train = x_train / 255
x_train = x_train.reshape((m, 28, 28, 1))
test = test / 255
test = test.reshape((-1, 28, 28, 1))

#One-hot output representaion
y_train = to_categorical(y_train, num_classes = 10)

# Split data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

print("Number of examples: ", m)
print("Training input shape: ", x_train.shape)
print("Training output shape: ", y_train.shape)
print("Validate input shape: ", x_val.shape)
print("Validate output shape: ", y_val.shape)
print("Test input shape: ", test.shape)
# Plot consecutive images
def sample_images(x,offset=0, sample_num=10):
    for i in range(sample_num):
        plt.subplot(math.ceil(sample_num/5), 5, i+1)
        plt.imshow(x[offset + i][:, :, 0])
    plt.show()

# Plot an image given it's index
def sample_image(x, index):
    plt.imshow(x[index][:, :, 0])


#sample_image(x_train, 1)
sample_images(x_train)
epochs = 20
batch_size = 64
learning_rate = 0.001
activation = 'tanh'
# Custom activations
def swish(x, beta=1):
    return (K.sigmoid(beta*x) * x)

def aria(x, alpha=1.25, beta=1):
    return ((K.sigmoid(beta*x)**alpha) * x)

get_custom_objects().update({'swish': swish, 'aria': aria})
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
datagen.fit(x_train)

batches = datagen.flow(x_train,y_train, batch_size=batch_size)
print("Number of training batches: ", len(batches))

first_batch = batches[0][0]
sample_images(first_batch, sample_num=25)
def model_1():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', 
                     activation=activation, input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(10, activation = "softmax"))
    model.summary()
    return model
#LeNet-5 like network
def model_2():
    model = Sequential()
    
    # Layer1
    model.add(Conv2D(filters=6, kernel_size=(5,5), activation=activation,
                     padding = 'same', input_shape = (28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    # Layer2
    model.add(Conv2D(filters=16, kernel_size=(5,5), activation=activation, 
                    padding = 'valid'))
    model.add(MaxPool2D(pool_size=(2,2)))
    #Layer3
    model.add(Flatten())
    model.add(Dense(120, activation=activation))
    #Layer4
    model.add(Dense(84, activation=activation))
    #Layer5
    model.add(Dense(10, activation = "softmax"))
    model.summary()
    return model
# LeNet5 variation
def model_3():
    model = Sequential()
    
    # Layer1
    model.add(Conv2D(filters=6, kernel_size=(5,5), activation=activation,
                     padding = 'same', input_shape = (28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    # Layer2
    model.add(Conv2D(filters=16, kernel_size=(5,5), activation=activation, 
                    padding = 'same'))
    model.add(MaxPool2D(pool_size=(2,2)))
    #Layer3
    model.add(Flatten())
    model.add(Dense(120, activation=activation))
    model.add(Dropout(0.25))
    #Layer4
    model.add(Dense(84, activation=activation))
    model.add(Dropout(0.25))
    #Layer5
    model.add(Dense(10, activation = "softmax"))
    model.summary()
    return model
optimizer = Adam(lr=learning_rate)
model = model_3()
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
fitted_model = model.fit_generator(generator=batches, validation_data = (x_val,y_val), 
                                   epochs = epochs, steps_per_epoch = m // batch_size)
# Evaluate model
# Loss plot
plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
fitted_model = model.fit(np.concatenate((x_train, x_val), axis=0), 
                         np.concatenate((y_train, y_val), axis=0), 
                         batch_size=batch_size, epochs=epochs)
ypred = model.predict(test)
ypred = np.argmax(ypred,axis=1)
submissions = pd.DataFrame({"ImageId": list(range(1,len(ypred)+1)),
                         "Label": ypred})
submissions.to_csv("cnn_model.csv", index=False, header=True)