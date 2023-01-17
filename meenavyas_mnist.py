# This is code is modification of 
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# Refer https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# Refer https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# https://www.learnopencv.com/image-classification-using-feedforward-neural-network-in-keras/
# Refer to its license for details
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K

print(check_output(["ls", "../input"]).decode("utf8"))
# we see that test.csv and train.csv are present
batch_size = 64
num_classes = 10 # 0 to 9
epochs = 20

df = pd.read_csv('../input/train.csv', sep=',') 
print(df.shape) # 42000, 785
#print(type(df)) # pandas.core.frame.DataFrame.
df.head(1)
# label, pixel0, pixel1, pixel2,...
# split into input x_train and output (y_train) variables
x = df.loc[:, df.columns != 'label'] # shape 42000, 785
y = df["label"] 
num_rows = x.shape[0]
input_dim = x.shape[1] # number of columns 784
print(input_dim) # 784
print(num_rows) # 42000
x = np.array(x)
# reshape into 28,28,1
x = x.reshape(num_rows,28,28,1)
x = np.array(x, dtype="float") / 255.0
# convert the labels from integers to vectors
y = keras.utils.to_categorical(y, num_classes)
from keras.layers.core import Activation, Flatten, Dense

model = Sequential()
model.add(Conv2D(20, (3, 3), padding="same", input_shape=(28, 28, 1))) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(50, (3, 3), padding="same", input_shape=(28, 28, 1))) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
# softmax is for single class predictions, sigmoid for multi class predictions
model.add(Dense(output_dim=num_classes))
model.add(Activation("softmax"))
# use binary_crossentropy if there are two classes
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# TBD learning rate, decay ??
#construct the image generator for data augmentation
import sys
print("Generating images...")
sys.stdout.flush()
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, \
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,\
    horizontal_flip=False, fill_mode="nearest")

# splitting data
print("splitting data into 75% and 25%...")
sys.stdout.flush()
from sklearn.model_selection import train_test_split
(x_train, valX, y_train, valY) = train_test_split(x,y,test_size=0.25, random_state=10)

# train the network
print("training network...")
sys.stdout.flush()

H = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size), \
    validation_data=(valX, valY), \
    validation_steps=len(valX) // batch_size, \
    steps_per_epoch=len(x_train) // batch_size, \
    epochs=epochs, verbose=1)
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
print("Saving model to disk")
import sys
sys.stdout.flush()
model.save("/tmp/mymodel")
df_test = pd.read_csv('../input/test.csv', sep=',') 
df_test.head(1) # pixel0..pixel783
test_num_records = df_test.shape[0]
print(test_num_records) # 28000
x_test = np.array(df_test)
x_test = x_test.reshape(test_num_records,28,28,1)
x_test = x_test.astype('float32')
x_test /= 255
from keras.models import load_model
mymodel = load_model('/tmp/mymodel')
yFit = mymodel.predict(x_test, batch_size=10, verbose=1)
#print(type(yFit)) # numpy.ndarray
import csv  
with open('output.csv', 'w', newline='') as csvfile:
    fieldnames = ['ImageId','Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for index in range(test_num_records): # 0 to test_num_records-1
        classesProbs = yFit[index]
        ansLabel = 0
        maxProb = 0;
        for idx in range(10): # 0 to 9
            if(classesProbs[idx] > maxProb):
                ansLabel = idx
                maxProb = classesProbs[idx]
        writer.writerow({'ImageId': index+1, 'Label': ansLabel})
print("Writing complete")