import numpy as np

from numpy import genfromtxt
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense

from keras.models import Model, Sequential

from keras.optimizers import Adam





from matplotlib.pyplot import imshow

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image



%matplotlib inline
def read(file='../input/fashion-mnist_train.csv', delimiter=','):

    '''

    Reads a csv file with given delimiter and returns data, labels (both np arrays)

    '''

    data = genfromtxt(file, skip_header=1, dtype=np.int64, delimiter=delimiter)

    return data[:, 1:], data[:, 0]





def reshape_xy(data_x, data_y):

    data_x = data_x.reshape(-1, 28, 28)

    data_y = convert_to_one_hot(data_y, classes).T

    return data_x/255., data_y





def split(data_x, data_y, val_percentage=0.2):

    perms = np.random.permutation(data_x.shape[0])

    training_indices = perms[int(perms.shape[0] * val_percentage):]

    val_indices = perms[:int(perms.shape[0] * val_percentage)]

    return data_x[training_indices], data_y[training_indices], data_x[val_indices], data_y[val_indices]





def load_dataset():

    data_x, data_y = reshape_xy(*read(file='../input/fashion-mnist_train.csv'))

    train_x, train_y, val_x, val_y = split(data_x, data_y, val_percentage=0.2)

    test_x, test_y = reshape_xy(*read(file='../input/fashion-mnist_test.csv'))

    return train_x, train_y,  val_x, val_y, test_x, test_y
# Flips images and adds it to the dataset passed [number of flips are chosen randomly]

def random_flips(data_x, data_y):

    num_flips = np.random.randint(int(0.5*data_x.shape[0]), data_x.shape[0])    

    perms = np.random.permutation(num_flips)

    flipped_x = np.flip(data_x[perms], 2)

    flipped_y = data_y[perms]

    return flipped_x, flipped_y                                  





# We flip a random number of images and and augment it to training data set

def augment_training_data(data_x, data_y):  

    flipped_x, flipped_y = random_flips(data_x, data_y)

    data_x = np.concatenate((data_x, flipped_x))

    data_y = np.concatenate((data_y, flipped_y))

    return data_x, data_y





def load_augmented_dataset():

    train_x, train_y,  val_x, val_y, test_x, test_y = load_dataset()

    train_x, train_y = augment_training_data(train_x, train_y)

    return train_x, train_y,  val_x, val_y, test_x, test_y





def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y    

    
def fashion_model(input_shape, classes=10):

    model = Sequential()

        

    # Stage 2

    model.add(Convolution1D(filters=32,  kernel_size=3, padding='same', input_shape=input_shape))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size=2))    

    model.add(Dropout(0.2))

    

    # Stage 3

    model.add(Convolution1D(filters=64,  kernel_size=3, padding='valid'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size=2))    

    model.add(Dropout(0.2))

        

    # Stage 4

    model.add(Convolution1D(filters=128,  kernel_size=3, padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size=2))    

    model.add(Dropout(0.25))

    

    model.add(Flatten())

    

    # Dense layer connected to 'Softmax' output

    model.add(Dense(256, activation='relu', name='fc_'+str(1)))

    

    # Dense layer connected to 'Softmax' output

    model.add(Dense(128, activation='relu', name='fc_'+str(2)))

    

    # Dense layer connected to 'Softmax' output

    model.add(Dense(classes, activation='softmax', name='fc_'+str(classes)))

    return model
classes = 10
# Let us load the data set

train_x, train_y,  val_x, val_y, test_x, test_y = load_augmented_dataset()
model = fashion_model(train_x[0].shape, classes)
layers = [(layer.input, layer.output) for layer in model.layers]
layers
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fit the model - with batch_size = 64, epochs = 25

model.fit(x=train_x, y=train_y, batch_size=128, epochs=250)
# Evaluate the model

print("validation set accuracy")

metrics = model.evaluate(x=val_x, y=val_y, batch_size=128)



print('Metrics: {m}'.format(m=metrics))
# Evaluate the model

print("test set accuracy")

metrics = model.evaluate(x=test_x, y=test_y, batch_size=128)



print('Metrics: {m}'.format(m=metrics))
model.summary()