import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adadelta, Adamax
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
base_path = '../input/fruits/fruits-360/'
test_path = base_path + 'Test/'
train_path = base_path + 'Training/'
categorys = os.listdir(train_path)
all_train = []
for i in categorys:
    for j in os.listdir(train_path + i + '/'):
        all_train.append(j)
all_test = []
for i in os.listdir(test_path):
    for j in os.listdir(test_path + i + '/'):
        all_test.append(j)
print('The training files:', len(all_train))
print('The testing files:', len(all_test))
train_generator = ImageDataGenerator(rescale=1.0/255)
test_generator = ImageDataGenerator(rescale=1.0/255, validation_split=0.7)
batch = 50
shape = 62
in_shape = (shape, shape, 3)
learn_r = 0.001
epoch = 10
activ = ['relu', 'swish']

train_datagen = train_generator.flow_from_directory(train_path,target_size=(shape, shape),
    color_mode="rgb", class_mode="categorical",batch_size=batch,shuffle=True)
test_datagen = test_generator.flow_from_directory(test_path, target_size=(shape, shape),
    color_mode="rgb", class_mode="categorical",batch_size=batch,shuffle=True, subset='training')
valid_datagen = test_generator.flow_from_directory(test_path, target_size=(shape, shape),
    color_mode="rgb", class_mode="categorical",batch_size=batch,shuffle=True, subset= 'validation')
model = tf.keras.models.Sequential([
        tf.keras.layers.Input(in_shape),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding='same', activation=activ[0]),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(4096, activation=activ[0]),
        tf.keras.layers.Dense(1000, activation=activ[0]),
        tf.keras.layers.Dense(1000, activation=activ[0]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(valid_datagen.num_classes, activation='softmax')
    ])
def get_nodel():
    base_model =  EfficientNetB4(input_shape=in_shape, weights=None, include_top=False, pooling='avg')
    x = base_model.output
    x = BatchNormalization()(x)
    predictions = Dense(valid_datagen.num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)
model1 = get_nodel()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.2,
                                                 patience=5, min_lr=0.0001)

opt_1 = SGD(learning_rate=learn_r, momentum=0.9)

opt_2 = Adam(learning_rate= learn_r)
model1.compile(optimizer= opt_2, loss= 'categorical_crossentropy', metrics= 'accuracy')
history = model1.fit(train_datagen, validation_data=valid_datagen, epochs=epoch, callbacks=reduce_lr)
pd.DataFrame(history.history).plot(figsize=(18, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
