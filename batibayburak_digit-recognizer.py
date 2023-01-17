import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline







np.random.seed(2)   #the random block of the validation set data to be always the same



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 
Y_train.head()
X_train.head()
X_train.isnull().any().describe()
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)
random_seed = 4
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

model
g = plt.imshow(X_train[0][:,:,0])
g = plt.imshow(X_train[1][:,:,0])
g = plt.imshow(X_train[2][:,:,0])
g = plt.imshow(X_train[3][:,:,0])
g = plt.imshow(X_train[4][:,:,0])
g = plt.imshow(X_train[5][:,:,0])
g = plt.imshow(X_train[6][:,:,0])
g = plt.imshow(X_train[7][:,:,0])
g = plt.imshow(X_train[8][:,:,0])
g = plt.imshow(X_train[9][:,:,0])
g = plt.imshow(X_train[10][:,:,0])
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



model.compile
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



epochs = 1 

batch_size = 86
data_overfitting = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False, 

        rotation_range=10,  

        zoom_range = 0.1, 

        width_shift_range=0.1, 

        height_shift_range=0.1, 

        horizontal_flip=False, 

        vertical_flip=False)



data_overfitting.fit(X_train)
data_overfitting
fitmodel = model.fit_generator(data_overfitting.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

results