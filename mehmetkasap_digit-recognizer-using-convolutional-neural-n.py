# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
print('train data shape: ', train.shape)
train.head()
train_data = train.iloc[:, 1:]
train_data.head()
train_label = train.iloc[:,0]
train_label.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data,train_label, test_size=0.1, random_state=42)
print('the number on the left shows # of imgaes, the number on the right shows # of pixels')
print('x_tain has shape: ', x_train.shape)
print('x_test has shape: ', x_test.shape)
print()
print('number of labels')
print('y_tain has shape: ', y_train.shape)
print('y_test has shape: ', y_test.shape)
# visualize number of digits classes (or labels)
plt.figure(figsize=(15,7))
sns.countplot(y_train, palette="icefire")
plt.title("Number of digit classes")
y_train.value_counts()
# plot some samples
img1 = x_train.iloc[1,:].as_matrix()
img1 = img1.reshape((28,28))
plt.imshow(img1,cmap='gray')
plt.title(y_train.iloc[1])
plt.axis("off")
plt.show()
# plot some samples
img1 = x_train.iloc[3,:].as_matrix()
img1 = img1.reshape((28,28))
plt.imshow(img1,cmap='gray')
plt.title(y_train.iloc[3])
plt.axis("off")
plt.show()
# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0
print("x_train shape: ",x_train.shape)
print("x_test shape: ",x_test.shape)
# plot some samples
img1 = x_train.iloc[1,:].as_matrix()
img1 = img1.reshape((28,28))
plt.imshow(img1,cmap='gray')
plt.title(y_train.iloc[1])
plt.axis("off")
plt.show()
# plot some samples
img1 = x_train.iloc[3,:].as_matrix()
img1 = img1.reshape((28,28))
plt.imshow(img1,cmap='gray')
plt.title(y_train.iloc[3])
plt.axis("off")
plt.show()
x_train.shape
# Reshape
# x_train = x_train.values.reshape(33600,28,28,1)
x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)
print("x_train shape: ",x_train.shape)
print("x_test shape: ",x_test.shape)
# x_train: 80 images, 28x28 pixels in each image, 1 dimensional color (grayscale)
# x_test: 20 images, 28x28 pixels in each image, 1 dimensional color (grayscale)
# Label Encoding 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)
y_train[0] # it is 8
y_train[10] # it is 6
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

# 1. filter & 2. filter + maxpool + dropout

model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 3. filter & 4. filter + maxpool + dropout

model.add(Conv2D(filters = 8, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 8, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# 5. filter & 6. filter + maxpool + dropout

# model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
#                  activation ='relu'))
# model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
#                  activation ='relu'))
# model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# model.add(Dropout(0.25))


# fully connected (ANN)

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# The activation is ‘softmax’. Softmax makes the output sum up to 1 so 
# the output can be interpreted as probabilities. 
# The model will then make its prediction based on which option has the highest probability.
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

# Define the optimizer
# adam: adaptive momentum
# lr: learning rate
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999) # for epoch>20 almost no change
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
#                                             patience=3, 
#                                             verbose=1, 
#                                             factor=0.5, 
#                                             min_lr=0.0001)
# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=10,  # randomly rotate images in the range 10 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
epochs = 20  # for better result increase the epochs
batch_size = 64

# Fit the model
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test), steps_per_epoch=x_train.shape[0] // batch_size)

# Without data augmentation
#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
#          validation_data = (X_val, Y_val), verbose = 2)
# Plot the loss curve for training and validation 
plt.plot(history.history['val_loss'], color='r', label="validation loss")
plt.title("Validation Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
# Plot the accuracy curves for training and validation 
plt.plot(history.history['val_acc'], color='g', label="validation accuracy")
plt.title("Validation Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
print('-'*80)
print('train accuracy of the model: ', history.history['acc'][-1])
print('-'*80)
print('-'*80)
print('validation accuracy of the model: ', history.history['val_acc'][-1])
print('-'*80)
test_data = pd.read_csv('../input/test.csv')
test_data = test_data.values.reshape(-1,28,28,1)
test_data.shape
# predict results
results = model.predict(test_data)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("Digit_Recognizer_Mechmet_Kasap.csv",index=False)