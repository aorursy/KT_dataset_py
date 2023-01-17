import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, Input, load_model

from keras.layers import Activation, Concatenate, AveragePooling2D, GlobalAveragePooling2D

from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

from keras.layers import Dropout, BatchNormalization

from keras.layers.convolutional import UpSampling2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from keras.initializers import RandomNormal

from keras.applications import DenseNet121

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Flatten

from keras.optimizers import RMSprop



sns.set(style='white', context='notebook', palette='deep')
### Kaggle or Local-PC ###

KAGGLE = True       # <==== SET ============

if KAGGLE:

    DIR = '../input'

else:               # local PC

    DIR = './'
# Load the data

train = pd.read_csv(os.path.join(DIR, "train.csv"))

test = pd.read_csv(os.path.join(DIR, "test.csv"))
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"], axis = 1) 



# free some space

del train
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0
SIZE = 28

N_ch = 1
# Reshape image in 3 dimensions (height, width, canal)

X_train = X_train.values.reshape(-1,SIZE,SIZE,N_ch)

test = test.values.reshape(-1,SIZE,SIZE,N_ch)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(

    X_train, Y_train, test_size = 0.1, random_state = 42)
# Dense block

def dense_block(input_tensor, input_channels, n_block, growth_rate):

    x = input_tensor

    n_ch = input_channels

    for i in range(n_block):

        # main root

        main = x

        # DenseBlock root

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        # Bottle-Neck 1x1 Convolution

        x = Conv2D(128, (1, 1))(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        # 3x3 Convolution

        x = Conv2D(growth_rate, (3, 3), padding="same")(x)

        # Concatenate

        x = Concatenate()([main, x])

        n_ch += growth_rate

    return x, n_ch
# Transition Layer

def transition_layer(input_tensor, input_channels, compression):

    n_ch = int(input_channels * compression)

    # 1x1 compression

    x = Conv2D(n_ch, (1, 1))(input_tensor)

    # AveragePooling

    x = AveragePooling2D((2, 2))(x)

    return x, n_ch
def build_denseNet(input_shape, growth_rate, blocks, compression):

    input = Input(shape = input_shape)



    n = growth_rate

    x = Conv2D(n, (1,1))(input)



    for i in range(len(blocks)):

        # Transition layer

        if i != 0:

            x, n = transition_layer(x, n, compression)

        # DenseBlock

        x, n = dense_block(x, n, blocks[i], growth_rate)



    x = GlobalAveragePooling2D()(x)

#     x = Dense(512, activation="relu")(x)

    output = Dense(10, activation = "softmax")(x)



    # model

    model = Model(input,output)



    return model
model = build_denseNet(input_shape = (SIZE, SIZE, N_ch),

                       growth_rate = 16,    # 32

                       blocks = [2,4,8,5],  # [2,4,8,5],[3,6,12,8],[6,12,24,16]

                       compression = 0.5)

# compile

model.compile(optimizer = Adam(lr=0.001),

              loss = "categorical_crossentropy",

              metrics = ["accuracy"])

    

model.summary()
#Callback : 



# Set a learning rate annealer

reduceLR = ReduceLROnPlateau(monitor = 'val_loss', 

                             patience = 2,

                             factor = 0.5,

                             min_lr = 1e-5,

                             verbose = 1)

# Save best model

chkPoint = ModelCheckpoint('dense.h5',

                           monitor = 'val_accuracy',

                           save_best_only = True,

                           save_weights_only = False,

                           mode = 'auto',

                           period = 1,

                           verbose = 1)

# Early stop

earlyStop = EarlyStopping(monitor = 'val_accuracy',

                          mode = 'auto',

                          patience = 5,

                          verbose = 1,

                          min_delta = 0)
epochs = 30

batch_size = 64     # 86
# With data augmentation to prevent overfitting (accuracy 0.99286)

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



datagen.fit(X_train)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train,

                                               batch_size=batch_size),

                                  epochs = epochs,

                                  validation_data = (X_val,Y_val),

                                  verbose = 1,

                                  steps_per_epoch=X_train.shape[0] // batch_size,

                                  callbacks=[reduceLR, chkPoint, earlyStop])

# Save hist

df = pd.DataFrame(history.history)

df.to_csv('hist.csv', index=False)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(df['loss'], color='b', label="Training loss")

ax[0].plot(df['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(df['accuracy'], color='b', label="Training accuracy")

ax[1].plot(df['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# predict results

results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.head()
submission.to_csv("cnn_mnist_datagen.csv",index=False)