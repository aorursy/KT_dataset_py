from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
batch_size = 128 #batch of images 
num_classes = 10 
epochs = 12 #no. of ephochs for training the data

# input image dimensions
img_rows, img_cols = 28, 28

#read data from csv files
train = pd.read_csv('../input/mnist_train.csv') #training split
val = pd.read_csv('../input/mnist_test.csv') #validation split
print(train.head()) #show data
# Assign features and corresponding labels
y_train = train['label'].values
x_train = train.iloc[:, 1:].values
y_val = val['label'].values
x_val = val.iloc[:, 1:].values
#reshape the features in a 28x28 matrix
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

#To reduce computational power convert into 0-1 range
x_train /= 255
x_val /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
#CNN model for training
model = Sequential() #sequential model (each layer acts on top of the other)

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1, 
          validation_data=(x_val, y_val))
#Compute final Validation loss and Accuracy
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])