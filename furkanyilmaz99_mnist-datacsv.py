# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import warnings
import warnings
#filter warnings 
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# read train
train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
print(train.shape)
train.head()
# read test
test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")
print(test.shape)
test.head()
# put labels into y_train variable 
Y_train = train["label"]
# Drop "label" column
X_train = train.drop(labels=["label"],axis=1)
# visualize number of digits  classes
plt.figure(figsize=(15,7))
g = sns.countplot(Y_train,palette="icefire")
plt.title("Number of digit classes")
Y_train.value_counts()
# put labels into y_test variable 
Y_test = test["label"]
# Drop "label" column
X_test = test.drop(labels=["label"],axis=1)
# visualize number of digits  classes
plt.figure(figsize=(15,7))
g = sns.countplot(Y_test,palette="icefire")
plt.title("Number of digit classes")
Y_test.value_counts()
# plot same samples 
img = X_train.iloc[1].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap="gray")
plt.title(train.iloc[1,0])
plt.axis("off")
plt.show()
# plot same samples 
img = X_train.iloc[0].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap="gray")
plt.title(train.iloc[0,0])
plt.axis("off")
plt.show()
# Normalize the data (datamızı 0 ile 1 arasına sıkıştırmaya normalize etmek denır)
# bir resmim maximium 255 değerini aldıgı için bizde resimleri 255 boleriz boyle 0,1 arasında olur
X_train = X_train/255
X_test  = X_test/255
print("X_train.shape : ",X_train.shape)
print("X_test.shape : ", X_test.shape)
#Reshape
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
print("X_train.shape : ",X_train.shape)
print("X_test.shape : ", X_test.shape)
# Label Encoding 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
Y_train = to_categorical(Y_train,num_classes=10)
Y_test = to_categorical(Y_test,num_classes=10)
# 
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
model = Sequential()
#
model.add(Conv2D(filters = 27, kernel_size = (3,3), padding="Same", activation="relu" ,input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.19))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 30, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# fully connected
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0,55))
model.add(Dense(10,activation="softmax"))
# Define the optimizer 
optimizer = Adam(lr = 0.001,beta_1=0.9,beta_2=0.999)
# compile the  model
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 55
batch_size=255
# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.6,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.6,  # randomly shift images horizontally 5%
        height_shift_range=0.7,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)#en sonunda elde ettiğimiz dataları datalarımızn oldugu yere fit ediyoruz
# Fit the model 
history = model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),epochs=epochs,validation_data=(X_test,Y_test),steps_per_epoch=X_train.shape[0]//batch_size)
# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
