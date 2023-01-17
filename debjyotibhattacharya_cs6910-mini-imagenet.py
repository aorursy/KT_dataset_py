from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2) 

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
training_set = train_datagen.flow_from_directory('../input/cs6910/4/train',batch_size=16, target_size=(224,224),
                                                  shuffle=True,
                                                  class_mode = "categorical")

test_datagen = ImageDataGenerator(rescale = 1./255) 
test_set = test_datagen.flow_from_directory('../input/cs6910/4/test',batch_size=16, target_size=(224,224),
                                            shuffle=False,
                                            class_mode = "categorical")
val_set=test_datagen.flow_from_directory('../input/cs6910/4/val', batch_size=16, 
                                            target_size=(224,224),
                                            shuffle=False,
                                            class_mode = "categorical")
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2) 

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout,MaxPooling2D,Convolution2D,Activation
import tensorflow as tf
from keras.regularizers import l2
import keras
from tensorflow.keras import datasets, layers, models
from keras.constraints import maxnorm
from keras.optimizers import SGD
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224,224,3)))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(33,activation='softmax'))
model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics=['accuracy'])
model.summary()
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
# Training the CNN on the Training set and evaluating it on the Test set
history=model.fit(x = training_set, validation_data =val_set, epochs =40)
model.save("cs6910percentaccuracy.h5")
model.evaluate(test_set)
model.save("55percentcs6910.h5")