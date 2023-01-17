# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau # Reduce learning rate when a metric has stopped improving.
from sklearn.model_selection import train_test_split
# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head() 
train.info()
test.head()
test.info()
X_train=train.drop('label',axis=1)
y_train=train['label']
sns.countplot(x=y_train)
y_train.value_counts()
# checking for null values
X_train.isnull().any().value_counts()
test.isnull().any().value_counts()
print('max value in one digit matrix',X_train.iloc[0].max())
print('min value in one digit matrix' , X_train.iloc[1].min())
#Normalization the data
X_train = X_train / 255.0
test = test / 255.0
X_train.shape
# reshapping image in three dimension
X_train = X_train.values.reshape(-1,28,28,1) # -1 it is an unknown dimension and we want numpy to figure it out.it is length of array
test = test.values.reshape(-1,28,28,1)
print('shape of full data', X_train.shape)
print('shape of one digit', X_train[0].shape)
# applying ONE-HOT-ENCODING
y_train=to_categorical(y_train,num_classes=10)
# splitting the data into training and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
#initialzing the model
model=Sequential()

# first Convolution layer
model.add(Conv2D(filters=32,kernel_size=(5,5),padding='Same',activation='relu',input_shape=(28,28,1)))

# Second Convoultion layer
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

#polling the layers
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# third Convolution layer
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
# fourth Convoultion layer
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

# Second polling layer
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# flattening the output from Convolution and polling layers
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
# Define the optimizer
optimizer = RMSprop()
# compling the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# decreases the learning rate during training
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
                                            factor=0.5, min_lr=0.00001)
# vertical_flip nor horizontal_flip since it could have lead to misclassify symetrical numbers such as 6 and 9.
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train) #Only required if featurewise_center or featurewise_std_normalization or zca_whitening are set to True.
batch_size=30
nb_train=len(X_train)
nb_val=len(X_val)
# Fit the model
history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = 20,verbose = 1, 
                              steps_per_epoch=nb_train // batch_size, 
                              validation_data = (X_val,y_val),
                              validation_steps=nb_val//batch_size
                              ,callbacks=[learning_rate_reduction])
model.save('Digit_recognizer.h5')
model.save_weights('Weight_file_for_Digit_recognizer.h5')
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
prob_pred= model.predict(test)

# select the indix with the maximum probability
results = np.argmax(prob_pred,axis = 1)
for i in range(10):
    plt.imshow(test[i].reshape(28,28),cmap='gray')
    plt.title('Predicted {}'.format(results[i]))
    plt.show()
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("my_submission.csv",index=False)
