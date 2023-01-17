

!pip install livelossplot

from livelossplot.tf_keras import PlotLossesCallback

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf



from sklearn.metrics import confusion_matrix

from sklearn import metrics



import numpy as np

np.random.seed(42)

import warnings;warnings.simplefilter('ignore')

%matplotlib inline

print('Tensorflow version:', tf.__version__)
train_images = pd.read_csv('../input/seti2ddata/train/train/images.csv', header=None)

# There is an error in the training label csv file existing in seti2ddata available on Kaggle. So, uploaded the correct training label

train_labels = pd.read_csv('../input/seti2ddata-correct/train_labels.csv', header=None) 



val_images = pd.read_csv('../input/seti2ddata/validation/validation/images.csv', header=None)

val_labels = pd.read_csv('../input/seti2ddata/validation/validation/labels.csv', header=None)
train_images.head()
train_labels.head()
val_images.head()
val_labels.head()
print("Training set shape:", train_images.shape, train_labels.shape)

print("Validation set shape:", val_images.shape, val_labels.shape)
# reshape the data into a shape that fits with CNN



x_train = train_images.values.reshape(3200, 64, 128, 1)

x_val = val_images.values.reshape(800, 64, 128, 1)



y_train = train_labels.values

y_val = val_labels.values
# Take 3 training images randomely and draw them 

plt.figure(0, figsize=(12,12))

for i in range(1,4):

    plt.subplot(1,3,i)

    img = np.squeeze(x_train[np.random.randint(0, x_train.shape[0])]) # np.squeeze is used to delete non  useful dimension in (64,128,1) and transform the shape into (64,128) in order to fit into plt.imshow

    plt.xticks([])

    plt.yticks([])

    plt.imshow(img,cmap="gray")
# Data augmentation using ImageDataGenerator

# An input batch of images is presented to the ImageDataGenerator.

# The ImageDataGenerator transforms each image in the batch by a series of random translations, rotations, etc.

# The randomly transformed batch is then returned to the calling function.

#The ImageDataGenerator is not returning both the original data and the transformed data — the class only returns the randomly transformed data.

# Ref:https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/



from tensorflow.keras.preprocessing.image import ImageDataGenerator 



datagen_train = ImageDataGenerator(horizontal_flip=True)

datagen_train.fit(x_train)



datagen_val = ImageDataGenerator(horizontal_flip=True)

datagen_val.fit(x_val)
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D

from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D



from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.utils import plot_model
# Initialising the CNN

model = Sequential()



# 1st Convolution

model.add(Conv2D(32,(5,5), padding='same', input_shape=(64, 128,1)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# 2nd Convolution layer

model.add(Conv2D(64,(5,5), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# Flattening

model.add(Flatten())



# Fully connected layer

model.add(Dense(1024))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.4))



model.add(Dense(4, activation='softmax'))
# Here, we have a initial learning rate that is fixed during first 5 steps and decreases exponentially afterwards

initial_learning_rate = 0.005

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    initial_learning_rate,

    decay_steps=5,

    decay_rate=0.96,

    staircase=True)   



optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) # https://www.tutorialspoint.com/keras/keras_model_compilation.htm

model.summary()
# ModelCheckpoint callback is used in conjunction with training using model.fit() to save a model or weights (in a checkpoint file) at some interval, 

#so the model or weights can be loaded later to continue the training from the state saved.

# Ref: https://keras.io/api/callbacks/model_checkpoint/

# https://towardsdatascience.com/checkpointing-deep-learning-models-in-keras-a652570b8de6



checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_loss',

                             save_weights_only=True, mode='min', verbose=0)

# PlotLossesCallback() is not supported by Tensorflow 2.2. 

# So, this is not used in the project. But still kept it as this can be useful when we will have tensorflow 2.1

callbacks = [PlotLossesCallback(), checkpoint]#, reduce_lr] 

batch_size = 32

history = model.fit(

    datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),

    steps_per_epoch=len(x_train)//batch_size,

    validation_data = datagen_val.flow(x_val, y_val, batch_size=batch_size, shuffle=True),

    validation_steps = len(x_val)//batch_size,

    epochs=12,

    callbacks=checkpoint

)
model.evaluate(x_val, y_val)
from sklearn.metrics import confusion_matrix

from sklearn import metrics

import seaborn as sns



y_true = np.argmax(y_val, 1)

y_pred = np.argmax(model.predict(x_val), 1)

print(metrics.classification_report(y_true, y_pred))

print("Classification accuracy: %0.6f" % metrics.accuracy_score(y_true, y_pred))
# Values of Recall from above confusion table can be seen on the diagonal below

labels = ["squiggle", "narrowband", "noise", "narrowbanddrd"]



ax= plt.subplot()

sns.heatmap(metrics.confusion_matrix(y_true, y_pred, normalize='true'), annot=True, ax = ax, cmap=plt.cm.Blues); #annot=True to annotate cells



# labels, title and ticks

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);