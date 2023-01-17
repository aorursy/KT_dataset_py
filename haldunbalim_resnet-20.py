import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D, Input, Flatten
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import keras
train_df = pd.read_csv("../input/digit-recognizer/train.csv")
test_df = pd.read_csv("../input/digit-recognizer/test.csv")
trainY = train_df["label"]
trainX = train_df.drop(labels = ["label"],axis = 1).values
testX = test_df.values
trainY.value_counts()
sns.countplot(trainY)
# check shape of input
trainX.shape,testX.shape
# We need to reshape X as (num_samples, width, heigh, channel)
trainX = trainX.reshape(len(trainX),28,28,1)
testX = testX.reshape(len(testX),28,28,1)
trainX.shape,testX.shape
# check shape of train output
trainY.shape
# We need to one-hot encode trainY
trainY = np.eye(10)[trainY]
trainY.shape
# Normalize Data
trainX = trainX / 255
testX = testX / 255
# Split train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(trainX, trainY, test_size = 0.2)
# Reference: https://keras.io/examples/cifar10_resnet/
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-3))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
    y = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
model = resnet_v1(trainX.shape[1:], depth=20, num_classes=10)
optimizer = optimizers.RMSprop(lr=1e-3)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
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
epochs = 30
batch_size = 128

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              ,callbacks=[learning_rate_reduction])
def plot_hist(history,ax_obj,keyword):
    ax_obj.plot(history.history[keyword], color='b', label="Train")
    ax_obj.plot(history.history['val_' + keyword], color='r', label="Validation")
    legend = ax_obj.legend(loc='best', shadow=True)
    ax_obj.grid()
    ax_obj.set_ylabel(keyword.title())
    ax_obj.set_xlabel("Epoch")
    ax_obj.set_title(keyword.title())
    
def plot_loss(history,ax_obj):
    plot_hist(history,ax_obj,"loss")

def plot_accuracy(history,ax_obj):
    plot_hist(history,ax_obj,"accuracy")

fig = plt.figure(figsize=(15,10))
ax = fig.subplots(2,1)
plot_loss(history, ax[0])
plot_accuracy(history, ax[1])
plt.figure(figsize=(8,8))
val_preds = model.predict(X_val)
val_preds = np.argmax(val_preds,axis=-1)
Y_val_classes = np.argmax(Y_val,axis=-1)
cm = confusion_matrix(val_preds,Y_val_classes)
cm = cm / np.sum(cm,axis=-1)[:,np.newaxis]
ax = sns.heatmap(cm,annot=True)
ax.set(xlabel='Predicted', ylabel='Actual',title="Confusion Matrix")
results = model.predict(testX)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)
