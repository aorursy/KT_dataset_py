# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings 

warnings.filterwarnings("ignore")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Loading Data 

train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
print(train.shape)

train.head() #We need to seperate Label 
print(test.shape)

test.head()
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)
#Test data

Y_test = test["label"]

X_test = test.drop(labels = ["label"],axis= 1)
#Lets Visualize number of digits classes

print(Y_train.value_counts())

plt.figure(figsize=(15,9))

sns.countplot(Y_train,palette = "icefire")

plt.title("Number of Digits Label Pixels")
#plotting some of the samples  

plt.subplot(2,2,1)

img1 = X_train.iloc[0].to_numpy().reshape((28,28))

plt.imshow(img1,cmap='gray')

plt.subplot(2,2,2)

img2 = X_train.iloc[10].to_numpy().reshape((28,28))

plt.imshow(img2,cmap='gray')

plt.subplot(2,2,3)

img3 = X_train.iloc[98].to_numpy().reshape((28,28))

plt.imshow(img3,cmap='gray')

plt.subplot(2,2,4)

img4 = X_train.iloc[25].to_numpy().reshape((28,28))

plt.imshow(img4,cmap='gray')

plt.show()
#Let's Normalize the Data 

X_train = X_train.astype("float32")

X_test = X_test.astype("float32")

X_train = X_train / 255.0

X_test = X_test / 255.0

print("X_train shape : ",X_train.shape)

print("Test shape : ",X_test.shape)
#Reshape 

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

print("X_train shape : ",X_train.shape)

print("Test shape : ",X_test.shape)
#Label Encoding 

from keras.utils.np_utils import to_categorical 

Y_train = to_categorical(Y_train,num_classes = 10) #we got 10 labels 

Y_test = to_categorical(Y_test,num_classes = 10)
from sklearn.model_selection import train_test_split 

X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.10, random_state = 42)

print("X_train shape",X_train.shape)

print("X_val shape",X_val.shape)

print("Y_train shape",Y_train.shape)

print("Y_val shape",Y_val.shape)
from sklearn.metrics import confusion_matrix

import itertools 



from keras.utils.np_utils import to_categorical #Converting to one hot encoding 

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.layers.normalization import BatchNormalization



epochs = 15

batch_size = 250

num_classes = 10



model = Sequential()

#

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = "Same", 

                 activation ='relu', input_shape = (28,28, 1))) 

             

model.add(MaxPool2D(pool_size=(3,3)))                                    

model.add(Dropout(0.25)) 

#



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = "Same", 

                 activation ='relu'))



model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



# 

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = "Same",

                 activation ='relu'))









# fully connected

model.add(Flatten())

model.add(Dense(128, activation = "relu")) #Hidden layer 1

model.add(Dropout(0.3))

model.add(Dense(10, activation = "softmax"))
optimizer = Adam(lr=0.001, beta_1 = 0.9, beta_2= 0.999)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
#data augmentation 

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=0.1,  # randomly rotate images in the range 5 degrees

        zoom_range = 0.1, # Randomly zoom image 5%

        width_shift_range=0.1,  # randomly shift images horizontally 5%

        height_shift_range=0.1,  # randomly shift images vertically 5%

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)



datagen.fit(X_train)    
history = model.fit_generator(datagen.flow(X_train, Y_train, 

                                           batch_size = batch_size),

                                           epochs = epochs, 

                                           validation_data = (X_val,Y_val),

                             steps_per_epoch = X_train.shape[0] // batch_size)
score = model.evaluate(X_test,Y_test,verbose = 0)

print("Test Loss : ",score[0])

print("Test Accuracy : ",score[1])
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Validation Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# confusion matrix

import seaborn as sns

# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
import matplotlib.pyplot as plt

%matplotlib inline

accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))

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