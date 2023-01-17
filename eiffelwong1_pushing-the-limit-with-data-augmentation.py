import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#reading training file

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
NUM_CLASS = 10



#making one hot encoding for the label

label = data['label']

label_one_hot = np.zeros((label.size, NUM_CLASS), dtype = 'float32')

for i in range(label.size):

    label_one_hot[i,label[i]] = 1

#remove the label column, so the remaining 784 columns can form a 28*28 photo

del data['label']



#changing data from DataFrame object to a numpy array, cause I know numpy better :p

data = data.to_numpy()



#here it shows that data can be fitted into a uint8, so by using a smaller type we can speed up the process

print('min,max value for data',np.min(data), np.max(data)) 

data = data.astype('uint8')



#making data to 28*28 photo

data = data.reshape(-1,28,28,1)





#checking out data shape

print(' data shape: {}, {} \n one hot lable shape: {}, {}'.format(

    data.shape, data.dtype, label_one_hot.shape, label_one_hot.dtype))
#spliting training and testing data sets

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data, label_one_hot, test_size = 0.1)
#data augmentation



from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
#simple CNN model with Keras

import keras

from keras.models import Model, Sequential

from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Dropout

from keras.layers import Dense, Flatten, Activation



model = Sequential([

    

    Convolution2D(filters = 16, kernel_size = (3,3), padding = "same", activation = 'relu', input_shape=(28,28,1)),

    Convolution2D(filters = 16, kernel_size = (3,3), padding = "same", activation = 'relu'),

    Convolution2D(filters = 32, kernel_size = (3,3), padding = "same", activation = 'relu'),

    Convolution2D(filters = 32, kernel_size = (3,3), padding = "same", activation = 'relu'),

    MaxPooling2D(pool_size=(2,2)),

    BatchNormalization(),

    Dropout(0.1),

    

    Convolution2D(filters = 64, kernel_size = (3,3), padding = "same", activation = 'relu'),

    Convolution2D(filters = 64, kernel_size = (3,3), padding = "same", activation = 'relu'),

    MaxPooling2D(pool_size=(2,2)),

    BatchNormalization(),

    Dropout(0.1),

    

    Flatten(),

    Dense(250, activation = 'relu'),

    Dense(10, activation = 'softmax'),

])
model.summary()
model.compile('adam',

              loss='categorical_crossentropy',

              metrics=['accuracy']

             )
#Starts training the model over the data 10 times.

#Here nothing fancy added for keeping it really really simple.



from keras.callbacks import EarlyStopping, ModelCheckpoint



CHECKPOINT_PATH = 'best.hdf5'



es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)

cp = ModelCheckpoint(CHECKPOINT_PATH, monitor = 'val_loss', mode = 'min', verbose = 0, save_best_only = True)



history = model.fit(datagen.flow(X_train, Y_train), validation_data = (X_test, Y_test), epochs = 1000, verbose = 1, callbacks = [es, cp])
import matplotlib.pyplot as plt



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.show()
#we read the csv before, but just read it again here.

val_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



#the same way to process the training data after seperating the label

val_data = val_data.to_numpy()

val_data = val_data.reshape(-1,28,28,1)



#here we ask the model to predict what the class is

model.load_weights(CHECKPOINT_PATH)

raw_result = model.predict(val_data)



#note: model.predict will return the confidence level for all 10 class,

#      therefore we want to pick the most confident one and return it as the final prediction

result = np.argmax(raw_result, axis = 1)



#generating the output, remember to submit the result to the competition afterward for your final score.

submission = pd.DataFrame({'ImageId':range(1,len(val_data) + 1), 'Label':np.argmax(raw_result, axis = 1)})

submission.to_csv('SimpleCnnSubmission.csv', index=False)