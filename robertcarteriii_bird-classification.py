from IPython.core.display import display, HTML
display(HTML("<style>.container {width:95% !importnat; }</style>"))
#Loading Libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, Flatten, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from IPython.display import display # Library to help view images
from PIL import Image # Library to help view images
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Library for data augmentation
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import ResNet50, InceptionV3, ResNet152

np.random.seed(42)

# Specify the traning, validation, and test dirrectories.
from fastai.vision import Path, ImageList, get_transforms, imagenet_stats

source_dir = Path('../input/100-bird-species/')
source_dir.ls()
train_dir = os.path.join(source_dir, 'train')
valid_dir = os.path.join(source_dir, 'valid')
test_dir = os.path.join(source_dir, 'test')

img_src = (ImageList.from_folder(source_dir).split_by_folder(train='train',valid='valid')
           .label_from_folder().add_test_folder('test').transform(get_transforms(), size=224))

bird_data = img_src.databunch(bs=32).normalize(imagenet_stats)
bird_data.show_batch()
#Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size = (224,224))

validation_generator = train_datagen.flow_from_directory(valid_dir, target_size = (224, 224))

test_generator = test_datagen.flow_from_directory(test_dir,target_size = (224, 224))
backend.clear_session()

resnetModel = ResNet152(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))

resnetModel.trainable = False 

resnet_train = Sequential()
resnet_train.add(resnetModel)
resnet_train.add(keras.layers.Flatten())
resnet_train.add(keras.layers.Flatten())
resnet_train.add(Dense(128, activation = 'relu'))
resnet_train.add(Dense(1, activation = 'sigmoid'))

resnet_train.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

history = resnet_train.fit_generator(
    train_generator,
    steps_per_epoch=500,
    epochs=50,
    validation_data=validation_generator,
    verbose = 1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 4, restore_best_weights = True)])

#Plot Model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate test data.
test_loss, test_acc = resnet_train.evaluate_generator(test_generator, steps = 50)
print('ResNet_train_test_acc:', test_acc)


# Build the model(s) and print (plot) the model.

from tensorflow.keras.utils import plot_model

backend.clear_session()

visible = Input(shape = (224,224,3))

conv11 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(visible)

conv21 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(conv11)

conv31 = BatchNormalization()(conv21)

conv41 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(conv31)

conv51 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(conv31)
conv51 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(conv51)

conv61 = AveragePooling2D((2,2), padding = 'same', strides = 1)(conv31)
conv61 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(conv61)

conv71 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(conv31)
conv71 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(conv71)
conv71 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(conv71)

merge = Concatenate(axis=-1)([conv41,conv51, conv61, conv71])

batch2 = BatchNormalization()(merge)

sep2d = SeparableConv2D(32, (3,3), padding = 'same', activation = 'relu')(batch2)
sep2d = SeparableConv2D(32, (3,3), padding = 'same', activation = 'relu')(sep2d)
sep2d = BatchNormalization()(sep2d)
sep2d = MaxPooling2D((2,2), padding = 'same')(sep2d)
sep2d = Dropout(0.5)(sep2d)

conv81 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(sep2d)
conv81 = Conv2D(32, (3,3), padding = 'same', activation = 'relu')(conv81)

batch3 = BatchNormalization()(conv81)

flat = Flatten()(batch3)

batch4 = BatchNormalization()(flat)

hidden1 = Dense(64, activation='relu')(batch4)
drop = Dropout(0.5)(hidden1)
output = Dense(190, activation='softmax')(drop)

modelx = Model(inputs=visible, outputs = output)

plot_model(modelx)
#Compile, fit, plot, and assess.  
modelx.compile(optimizer = 'adam',
               loss = 'binary_crossentropy',
               metrics = ['accuracy'])

history = modelx.fit_generator(
    train_generator,
    steps_per_epoch=500,
    epochs=50,
    validation_data=validation_generator,
    verbose = 1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 4, restore_best_weights = True)])

#Plot Model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss values vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate test data.
test_loss, test_acc = modelx.evaluate_generator(test_generator, steps = 50)
print('ResNet_train_test_acc:', test_acc)
