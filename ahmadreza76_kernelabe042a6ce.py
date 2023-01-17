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
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.models import Sequential

from keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers

from keras.layers import Dense, Dropout, Flatten # core layers



from keras.layers.normalization import BatchNormalization



from keras.preprocessing.image import ImageDataGenerator



from keras.utils.np_utils import to_categorical
#load data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')

#print size of training data and testing data

print(f"Size of training data => {train.shape}\nSize of Testing data => {test.shape}")
#split labels and image data

X = train.drop(['label'], 1).values

y = train['label'].values



test_x = test.values
#convert our images to grayscale

X = X / 255.0

test_x = test_x / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

# canal = 1 => For gray scale

X = X.reshape(-1,28,28,1)

test_x = test_x.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y = to_categorical(y)



print(f"Label size {y.shape}")
# Split the train and the validation set for the fitting (90% train, 10% test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

#print new sizes

X_train.shape, X_test.shape, y_train.shape, y_test.shape
epochs = 50

batch_size = 64
#Defining model of our CNN

model=Sequential()



model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))

model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))



model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))



model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())    

model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))

    

model.add(MaxPooling2D(pool_size=(2,2)))

    

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(512,activation="relu"))

    

model.add(Dense(10,activation="softmax"))

    

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#In order to avoid overfitting problem, we need to expand artificially our handwritten digit dataset

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



train_gen = datagen.flow(X_train, y_train, batch_size=batch_size)

test_gen = datagen.flow(X_test, y_test, batch_size=batch_size)

#Randomly rotate some training images by 10 degrees

#Randomly Zoom by 10% some training images

#Randomly shift images horizontally by 10% of the width

#Randomly shift images vertically by 10% of the height

# Fit the model

history = model.fit_generator(train_gen, 

                              epochs = epochs, 

                              steps_per_epoch = X_train.shape[0] // batch_size,

                              validation_data = test_gen,

                              validation_steps = X_test.shape[0] // batch_size)




fig = plt.figure(figsize=(10, 10)) # Set Figure



y_pred = model.predict(X_test) # Predict encoded label as 2 => [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]



Y_pred = np.argmax(y_pred, 1) # Decode Predicted labels

Y_test = np.argmax(y_test, 1) # Decode labels



mat = confusion_matrix(Y_test, Y_pred) # Confusion matrix



# Plot Confusion matrix

sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)

plt.xlabel('Predicted Values')

plt.ylabel('True Values');

plt.show();



#Submition & Prediciting the Outputs

pred = model.predict_classes(test_x, verbose=1)
submission['Label'] = pred

submission.to_csv("CNN_keras_sub.csv", index=False)

submission.head()