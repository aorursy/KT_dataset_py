import os
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Lambda, Input, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU,ThresholdedReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam,Nadam,Adamax,TFOptimizer
#Import Data
data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

#from subprocess import check_output
#print(check_output(["ls", "../input/vgg16-weights-tf"]).decode("utf8"))
#../input/vgg16-weights-tf/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
# Set the parameter
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

X = np.array(data_train.iloc[:, 1:])
y = to_categorical(np.array(data_train.iloc[:, 0]))

# Split validation data to optimize classifier during training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

# Test data
X_test = np.array(data_test.iloc[:, 1:])
y_test = to_categorical(np.array(data_test.iloc[:, 0]))

# Train and Validation Data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

# Convert into format acceptable by tensorflow
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255
# Model Parameters
batch_size = 256
num_classes = y_test.shape[1]
epochs = 100

# Parameter for Image Data Augumentation
shift_fraction=0.005

# Create Model
fashion_model = Sequential()
fashion_model.add(Conv2D(32,kernel_size=(3, 3),activation='linear',input_shape=input_shape,padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.5))
fashion_model.add(Conv2D(64,(3, 3),activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.5))
fashion_model.add(Conv2D(128,(3, 3),activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.5))
fashion_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128,activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dense(num_classes,activation='softmax'))
fashion_model.compile(loss=keras.losses.categorical_crossentropy,optimizer=Adamax(),metrics=['accuracy'])
#fashion_model.summary()
# Train the model
#fashion_train = fashion_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))

gen = ImageDataGenerator(width_shift_range=shift_fraction,height_shift_range=shift_fraction,horizontal_flip=True)
batches = gen.flow(X_train, y_train, batch_size=batch_size)
val_batches = gen.flow(X_val, y_val, batch_size=batch_size)

fashion_train=fashion_model.fit_generator(batches, steps_per_epoch=X_train.shape[0]//batch_size, epochs=epochs,validation_data=val_batches, 
                                                   validation_steps=X_val.shape[0]//batch_size, use_multiprocessing=True)
# Evaluate Model against test data and get the score
score = fashion_model.evaluate(X_test, y_test, verbose=0)

# Print Metrics
print (score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Extract Metrics for plotting
accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))

# Plots
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
# Predict classes for the test data
predicted_classes = fashion_model.predict_classes(X_test)

# Get the Indices to be plotted
y_true = data_test.iloc[:, 0]
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]
print (incorrect)
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_true, predicted_classes, target_names=target_names))