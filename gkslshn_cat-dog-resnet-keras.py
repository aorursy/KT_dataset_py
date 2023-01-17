from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.model_selection import train_test_split

from keras.applications import VGG16

import matplotlib.pyplot as plt

from keras.optimizers import SGD



from keras import optimizers

from keras import models

from keras import layers





import pandas as pd

import numpy as np

import zipfile, os





with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as zip:

    zip.extractall()    

    zip.close()
filenames = os.listdir("/kaggle/working/train")



labels = []



for filename in filenames:

    label = filename.split(".")[0] 

    if label == "cat":

        labels.append("0")

    else:

        labels.append("1")

df = pd.DataFrame({"id": filenames, "label" : labels})



print(df.shape)

print(df.head())


# define the total number of epochs to train for along with the

# initial learning rate



NUM_EPOCHS = 50

INIT_LR = 1e-1



def poly_decay(epoch):

    # initialize the maximum number of epochs, base learning rate,

    # and power of the polynomial

    maxEpochs = NUM_EPOCHS

    baseLR = INIT_LR

    power = 1.0



    # compute the new learning rate based on polynomial decay

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power



    # return the new learning rate

    return alpha



from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import AveragePooling2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.convolutional import ZeroPadding2D

from keras.layers.core import Activation

from keras.layers.core import Dense

from keras.layers import Flatten

from keras.layers import Input

from keras.models import Model

from keras.layers import add

from keras.regularizers import l2



def residual_module(data, K, stride, chanDim, red=False,

    reg=0.0001, bnEps=2e-5, bnMom=0.9):



    # the shortcut branch of the ResNet module should be

    # initialize as the input (identity) data

    shortcut = data



    # the first block of the ResNet module are the 1x1 CONVs

    bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,

        momentum=bnMom)(data)

    act1 = Activation("relu")(bn1)



    conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,

        kernel_regularizer=l2(reg))(act1)



    # the second block of the ResNet module are the 3x3 CONVs

    bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,

        momentum=bnMom)(conv1)

    act2 = Activation("relu")(bn2)

    conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,

        padding="same", use_bias=False,

        kernel_regularizer=l2(reg))(act2)



    #the third block of the ResNet module is another set of 1x1 CONVs

    bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,

        momentum=bnMom)(conv2)

    act3 = Activation("relu")(bn3)

    conv3 = Conv2D(K, (1, 1), use_bias=False,

        kernel_regularizer=l2(reg))(act3)



    # if we are to reduce the spatial size, apply a CONV layer to

    # the shortcut

    if red:

        shortcut = Conv2D(K, (1, 1), strides=stride,

            use_bias=False, kernel_regularizer=l2(reg))(act1)

    

    # add together the shortcut and the final CONV

    x = add([conv3, shortcut])



    # return the addition as the output of the ResNet module

    return x



def build_model(width, height, depth, classes, stages, filters,

        reg=0.0001, bnEps=2e-5, bnMom=0.9):



        # initialize the input shape to be "channels last" and the

        # channels dimension itself

        inputShape = (height, width, depth)

        chanDim = -1



        # set the input and apply BN

        inputs = Input(shape=inputShape)

        x = BatchNormalization(axis=chanDim, epsilon=bnEps,

            momentum=bnMom)(inputs)



        # apply CONV => BN => ACT => POOL to reduce spatial size       

        x = Conv2D(filters[0], (5, 5), use_bias=False,

        padding="same", kernel_regularizer=l2(reg))(x)

        x = BatchNormalization(axis=chanDim, epsilon=bnEps,

        momentum=bnMom)(x)

        x = Activation("relu")(x)

        x = ZeroPadding2D((1, 1))(x)

        x = MaxPooling2D((3, 3), strides=(2, 2))(x)



        # loop over the number of stages

        for i in range(0, len(stages)):

            # initialize the stride, then apply a residual module

            # used to reduce the spatial size of the input volume

            stride = (1, 1) if i == 0 else (2, 2)

            x = residual_module(x, filters[i + 1], stride,

                chanDim, red=True, bnEps=bnEps, bnMom=bnMom)



            # loop over the number of layers in the stage

            for j in range(0, stages[i] - 1):

                # apply a ResNet module

                x = residual_module(x, filters[i + 1],

                    (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)



        # apply BN => ACT => POOL

        x = BatchNormalization(axis=chanDim, epsilon=bnEps,

        momentum=bnMom)(x)

        x = Activation("relu")(x)

        x = AveragePooling2D((8, 8))(x)





        # softmax classifier

        x = Flatten()(x)

        x = Dense(classes, kernel_regularizer=l2(reg))(x)

        x = Activation("softmax")(x)



        # create the model

        model = Model(inputs, x, name="resnet")



        # return the constructed network architecture

        return model



trainGen = ImageDataGenerator(rescale=1./255,

                                  rotation_range=40,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True,

                                  fill_mode='nearest')



testGen = ImageDataGenerator(rescale=1. / 255)





train_df, validation_df = train_test_split(df, test_size=0.25)



train_size = train_df.shape[0]



validation_size = validation_df.shape[0]



batch_size = 16



print("train size: " + str(train_size)+"\nvalidation_size:" + str(validation_size))


train_it = trainGen.flow_from_dataframe(train_df,

    "train/",

    x_col="id",

    y_col="label",

    #class_mode="binary",

    target_size=(200,200),

    batch_size = batch_size)





validation_it = trainGen.flow_from_dataframe(validation_df,

    "train/",

    x_col="id",

    y_col="label",

    #class_mode="binary",

    target_size=(200,200),

    batch_size = batch_size)

#compiling model...

opt = SGD(lr=INIT_LR, momentum=0.9)

opt="adam"

model = build_model(200, 200, 3, 2, (9, 9, 9),

    (64, 64, 128, 256), reg=0.0005)

model.compile(loss="binary_crossentropy", optimizer=opt,

    metrics=["accuracy"])



model.summary()
H = model.fit_generator(train_it,

    steps_per_epoch=train_size//batch_size,

    epochs=50,

    validation_data=validation_it,

    validation_steps=validation_size//batch_size)



model.save("catsvsdogs_resnet.hdf5")


plt.style.use('ggplot')



acc = H.history['accuracy']

val_acc = H.history['val_accuracy']

loss = H.history['loss']

val_loss = H.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'b', label='training acc')

plt.plot(epochs, val_acc, 'r', label='validation acc')

plt.title('accuracy')

plt.legend()



plt.figure()

plt.plot(epochs, loss, 'b', label='training loss')

plt.plot(epochs, val_loss, 'r', label='validation loss')

plt.title('loss')

plt.legend()



plt.show()