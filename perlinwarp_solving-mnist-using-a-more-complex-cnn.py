import numpy as np
import pandas as pd
from tensorflow.python import keras

img_rows, img_cols = 28, 28
num_classes = 10

#Prepping the training data 
train_file = "../input/train.csv"
data = np.loadtxt(train_file, skiprows=1, delimiter=',')
y = data[:, 0]
y = keras.utils.to_categorical(y, num_classes)

x = data[:,1:]
num_images = data.shape[0]
x = x.reshape(num_images, img_rows, img_cols, 1)
x = x / 255 # Scaling our data between 0 and 1 helps the adam optimiser pick the right learning rate

#Prepping the test data
test_file = "../input/test.csv"
test = np.loadtxt(test_file, skiprows=1, delimiter=',')
test = test / 255
num_images = test.shape[0]
test = test.reshape(num_images,img_rows, img_cols, 1)
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D

#Specifying Model Architecture
model = Sequential()
model.add(Conv2D(12, kernel_size=(3,3), activation='relu', input_shape=(img_rows,img_cols,1)))
model.add(Conv2D(12, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(12, kernel_size=(3,3), activation='relu'))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
#Fitting the model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

valdata = ImageDataGenerator()
valdata.fit(x)

datagen = ImageDataGenerator(
        zoom_range=0.1,
        rotation_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

datagen.fit(x)

model.fit_generator(datagen.flow(x,y,batch_size=100), epochs = 4, validation_data=valdata.flow(x,y))
results = model.predict(test)
results = np.argmax(results,axis=1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
