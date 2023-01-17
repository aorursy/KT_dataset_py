#Imports



import numpy as np

import pandas as pd

from IPython.display import Image

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, Activation

from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.optimizers import RMSprop, Adam

from tensorflow.python.keras.activations import relu

from tensorflow.python.keras.utils import plot_model

from tensorflow.python.keras.losses import categorical_crossentropy
#Read Data

digit_data = pd.read_csv('../input/train.csv')

digit_data.head(5)
img_rows, img_cols = 28,28

num_classes = 10



#Preparing the training data

def data_prep_train(raw,val_frac):

    """

    Prepares the training data for our model

    inputs:

        raw: raw dataset

        val_frac: integer between 0 and 1. Fraction of the data to be used for validation.

    

    Outputs:

        X_train: training examples

        X_val: validation examples

        y_train: training lables

        y_val: validation lables

    """

    num_images = int(raw.shape[0])    

    

    y_full = keras.utils.to_categorical(raw.label, num_classes)

    

    X_as_array = raw.values[:,1:]

    X_shaped_array = X_as_array.reshape(num_images, img_rows, img_cols, 1)

    X_full = X_shaped_array / 255

    

    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=val_frac)

    return X_train, X_val, y_train, y_val



#Preparing the test data

    """

    Prepares the test data for our model

    inputs:

        raw: raw dataset

    

    Outputs:

        X: test examples

    """

def data_prep_predict(raw):

    num_images = int(raw.shape[0])

    X_as_array = raw.values

    X_shaped_array = X_as_array.reshape(num_images, img_rows, img_cols, 1)

    X = X_shaped_array / 255

    return X
def build_model(layer_sizes=[32, 32, 64, 64, 256], kernel_sizes=[5,5,3,3], activation = 'relu'):

    """

    building a CNN with 4+2 layers.

    inputs:

        layer_sizes: list of length 5 containing the number of hidden units in each layer

        kernel_sizes: list of length 4 containing the size of the kernels in the Conv Layers

        activation: The activation function, string or function.

        

    output:

        model: The finished model

    """

    model = Sequential()

    

    model.add(Conv2D(layer_sizes[0], kernel_size=kernel_sizes[0], padding = 'same', input_shape=(img_rows, img_cols, 1)))

    model.add(BatchNormalization())

    model.add(Activation(activation))

    model.add(Conv2D(layer_sizes[1], kernel_size=kernel_sizes[1], padding = 'same'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(rate=0.25))



    model.add(Conv2D(layer_sizes[2], kernel_size=kernel_sizes[2], padding = 'same'))

    model.add(BatchNormalization())

    model.add(Activation(activation))

    model.add(Conv2D(layer_sizes[3], kernel_size=kernel_sizes[3], padding = 'same'))

    model.add(BatchNormalization())

    model.add(Activation(activation))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(rate=0.25))

    

    model.add(Flatten())

    model.add(Dense(layer_sizes[4]))

    model.add(BatchNormalization())

    model.add(Activation(activation))

    model.add(Dropout(rate=0.5))



    model.add(Dense(num_classes, activation='softmax'))

    

    return model



my_model = build_model()

plot_model(my_model, to_file='my_model.png', show_shapes=True, show_layer_names=True)

Image('my_model.png')
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

        vertical_flip=False,)  # randomly flip images
def train_model(model, optimizer='adam', batch_size=64, epochs=1, verbose=1, callbacks=[]):

    """

    Trains the model.

    Outputs:

        history: dictionary containing information about the training process like training and validation accuracy

    """

    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    

    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),

                            epochs=epochs,

                            verbose=verbose,

                            validation_data=(X_val,y_val),

                            callbacks=callbacks)

    return history
# leaky_relu = lambda x: relu(x, alpha=0.1)

# X_train, X_val, y_train, y_val = data_prep_train(digit_data,0.2)



# learning_rates = [0,0015,0.003,0.006]



# histories = []

# for lr in learning_rates:

#     optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#     lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.000001)

#     my_model = build_model(activation=leaky_relu)

#     histories.append(train_model(my_model, optimizer=optimizer, epochs=35, batch_size = 64, verbose=2, callbacks=[lr_reduction]))



# colors = ['red', 'blue', 'green', 'purple', 'grey', 'yellow']



# plt.figure(figsize=(20,9))

# for i, lr in enumerate(learning_rates):

#     plt.plot(range(25,36), histories[i].history['val_acc'][24:], color=colors[i],label='learning rate: '+str(lr))

# legend = plt.legend(loc='best', shadow=True)

# plt.show()

X_train, X_val, y_train, y_val = data_prep_train(digit_data,0.1)



leaky_relu = lambda x: relu(x, alpha=0.1)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

lr_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

my_model = build_model(activation=leaky_relu)

history = train_model(my_model, optimizer=optimizer, epochs=40, batch_size = 128, verbose=1, callbacks=[lr_reduction])



plt.figure(figsize=(20,9))

plt.plot(range(20,41),history.history['val_acc'][19:], color='red', label='validation accuracy')

plt.plot(range(20,41),history.history['acc'][19:], color='blue', label='accuracy')

legend = plt.legend(loc='best', shadow=True)

plt.show()
subm_examples = pd.read_csv('../input/test.csv')

X_subm = data_prep_predict(subm_examples)

y_subm = my_model.predict(X_subm)

n_rows = y_subm.shape[0]

y_subm = [np.argmax(y_subm[row,:]) for row in range(n_rows)]

output = pd.DataFrame({'ImageId': range(1,n_rows+1), 'Label': y_subm})

output.to_csv('submission.csv', index=False)