import numpy as np
import pandas as pd
import keras 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import Input, GlobalAveragePooling2D
from keras import models
from keras.models import Model
train = pd.read_csv("../input/digit-recognizer/train.csv")
train.head()
Y = train["label"]
X = train.drop(["label"], axis=1)
X = X.values.reshape(-1,28,28,1)
X = X/255.0
X.shape
train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
nRows,nCols,nDims = train_X.shape[1:]
input_shape = (nRows, nCols, nDims)
input_img = Input(shape=(28, 28, 1))
l1 = Conv2D(64, (3,3), activation='relu', padding = 'valid')(input_img)
l2 = Conv2D(32, (1,1), padding='valid', activation='relu')(l1)
l3 = MaxPooling2D((2,2),strides=(2,2))(l2)
mid = Conv2D(8, (3,3), padding='valid', activation='relu')(l3)
l4 = Conv2D(8, (1,1), padding='valid', activation='relu')(mid)
l5 = MaxPooling2D((2,2),strides=(1,1))(l4)
l6 = Conv2D(8, (3,3), padding='valid', activation='relu')(l5)
l7 = Conv2D(6, (1,1), padding='valid', activation='relu')(l6)
output = Flatten()(l7)
out    = Dense(10, activation='softmax')(output)
model = Model(inputs = input_img, outputs = out)
model.summary()
plot_model(model, to_file='model_plot_MNIST_9.5K.png', show_shapes=True, show_layer_names=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam",metrics=['accuracy'])
history = model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=20, batch_size=32)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()