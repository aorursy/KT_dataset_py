# Imports statements
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, Flatten, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import os
from fastai.vision import *
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
# Directory calls
source_dir = Path('../input/100-bird-species/')
source_dir.ls()
train_dir = os.path.join(source_dir, 'train')
valid_dir = os.path.join(source_dir, 'valid')
test_dir = os.path.join(source_dir, 'test')
img_src = (ImageList.from_folder(source_dir)
                .split_by_folder(train='train', valid='valid')
                .label_from_folder()
                .add_test_folder('test')
                .transform(get_transforms(), size=224))

bird_data = img_src.databunch(bs=32).normalize(imagenet_stats)
bird_data.show_batch()
# Create data sets
data_gen = ImageDataGenerator(rescale=1./255, )
train_data = data_gen.flow_from_directory(train_dir, target_size=(224,224))
valid_data = data_gen.flow_from_directory(valid_dir, target_size=(224,224))
test_data = data_gen.flow_from_directory(test_dir, target_size=(224,224))
# Build model
backend.clear_session()
conv_base = InceptionV3 (weights = 'imagenet', 
                  include_top = False,
                  input_shape = (224, 224, 3))
conv_base.trainable = False # Freeze the Inception V3 weights.

model = Sequential()
model.add(conv_base)
model.add(BatchNormalization())
model.add(keras.layers.Flatten())
model.add(Activation('relu'))
model.add(Dense(190))
model.add(Activation('softmax'))
# Compile model
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.001, decay=1e-6, momentum=0.9),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])
# Plot graphs
history = model.fit_generator(
    train_data,
    steps_per_epoch=766,
    epochs=50,
    validation_data=valid_data,
    validation_steps=29,
    verbose = 1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 4, restore_best_weights = True)])

#plot accuracy vs epoch
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
scores = model.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
backend.clear_session()

# Input model
input_a = Input(shape=(224,224,3))

# Create Tower 1
conv2d_a = Conv2D(64, (3,3), padding = 'same', activation='relu')(input_a)

# Create Tower 2
conv2d_b = Conv2D(64, (3,3), padding = 'same', activation='relu')(input_a)

# Batch Norm.
batch_normalization_a = BatchNormalization()(input_a)

# Create Tower 3
conv2d_c = Conv2D(64, (3,3), padding = 'same', activation='relu')(batch_normalization_a)
pool_a = AveragePooling2D((3, 3), padding = 'same', strides=(1,1))(conv2d_c)

# Create Tower 4
conv2d_d = Conv2D(64, (3,3), padding = 'same', activation='relu')(batch_normalization_a)
pool_b = AveragePooling2D((3, 3), padding = 'same', strides=(1,1))(conv2d_d)

# Create Tower 5
conv2d_e = Conv2D(64, (3,3), padding = 'same', activation='relu')(batch_normalization_a)
conv2d_f = Conv2D(64, (3,3), padding = 'same', activation='relu')(conv2d_e)
pool_c = AveragePooling2D((3, 3), padding = 'same', strides=(1,1))(conv2d_f)

# Create Tower 6
conv2d_g = Conv2D(64, (3,3), padding = 'same', activation='relu')(batch_normalization_a)
conv2d_h = Conv2D(64, (3,3), padding = 'same', activation='relu')(conv2d_g)
conv2d_i = Conv2D(64, (3,3), padding = 'same', activation='relu')(conv2d_h)
pool_d = AveragePooling2D((3, 3), padding = 'same', strides=(1,1))(conv2d_i)

# Concatentate
concatenate_a = Concatenate(axis=-1)([pool_a, pool_b, pool_c, pool_d])

# Batch Norm.
batch_normalization_b = BatchNormalization()(concatenate_a)

# Create Tower 7
conv2d_j = Conv2D(64, (3,3), strides=(1,1),padding = 'same', activation='relu')(batch_normalization_b)

# Create Tower 8
conv2d_k = Conv2D(64, (3,3), strides=(1,1), padding = 'same', activation='relu')(conv2d_j)

# Batch Norm.
batch_normalization_c = BatchNormalization()(conv2d_k)

# Flatten
flat_a = keras.layers.Flatten()(batch_normalization_c)

# Batch Norm.
batch_normalization_d = BatchNormalization()(flat_a)

# Hidden connected layer and output
dense_a = Dense(32, activation='relu')(batch_normalization_d)
dropout_a = Dropout(0.5)(dense_a)
dense_b = Dense(190, activation='softmax')(dropout_a)

model_2 = Model(inputs=input_a, outputs=dense_b)
# plot graph
plot_model(model_2)
# Compile, fit, plot, and assess
model_2.compile(optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

history = model_2.fit_generator(
    train_data,
    steps_per_epoch=766,
    epochs=50,
    validation_data=valid_data,
    validation_steps=29,
    verbose = 1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 4, restore_best_weights = True)])
#plot accuracy vs epoch
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
scores = model_2.evaluate(test_data, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])