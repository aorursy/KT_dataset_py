# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
# Read test data

test = pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
# Get digin labels to yTrain

yTrain = train["label"]



# Drop 'label' column

xTrain = train.drop(labels = ["label"], axis = 1) 
plt.figure(figsize = (15,7))

g = sns.countplot(yTrain, palette = "icefire")

plt.title("Number of digits")

yTrain.value_counts()
def drawImage(imgArray):    

    imgArray = imgArray.reshape((28, 28))

    plt.imshow(imgArray, cmap = 'gray')

    plt.title(train.iloc[0,0])

    plt.axis("off")

    plt.show()
# plot some samples

drawImage(xTrain.iloc[9].as_matrix())

drawImage(xTrain.iloc[5].as_matrix())
xTrain = xTrain / 255.0

test = test / 255.0

print("xTrain shape: ",xTrain.shape)

print("test shape: ",test.shape)
xTrain = xTrain.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("xTrain shape: ",xTrain.shape)

print("test shape: ",test.shape)
# one-hot-encoding

from keras.utils.np_utils import to_categorical

yTrain = to_categorical(yTrain, num_classes = 10)
from sklearn.model_selection import train_test_split

xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size = 0.1, random_state = 2)

print("xTrain shape",xTrain.shape)

print("xVal shape",xVal.shape)

print("yTrain shape",yTrain.shape)

print("yVal shape",yVal.shape)
# Draw an example

drawImage(xTrain[2][:,:,0])
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import LearningRateScheduler



model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (4,4),padding = 'Same', activation = 'relu', input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2), strides = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
optimizer = Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics = ["accuracy"])
epochs = 30 # 1 epoch means 1 forward and 1 backward pass.

batch_size = 385 # Number of training samples for one forward/backward pass.
datagen = ImageDataGenerator(

        featurewise_center = False,  # set input mean to 0 over the dataset

        samplewise_center = False,  # set each sample mean to 0

        featurewise_std_normalization = False,  # divide inputs by std of the dataset

        samplewise_std_normalization = False,  # divide each input by its std

        zca_whitening = False,  # dimesion reduction

        rotation_range = 10,  # randomly rotate images in the range 10 degrees

        zoom_range = 0.1, # Randomly zoom image 1%

        width_shift_range = 0.1,  # randomly shift images horizontally 1%

        height_shift_range = 0.1,  # randomly shift images vertically 1%

        horizontal_flip = False,  # randomly flip images

        vertical_flip = False)  # randomly flip images



datagen.fit(xTrain)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
# fit the model

history = model.fit_generator(datagen.flow(xTrain,

                                           yTrain, 

                                           batch_size = batch_size), 

                              epochs = epochs, 

                              validation_data = (xVal, yVal), 

                              steps_per_epoch = xTrain.shape[0] // batch_size,

                              callbacks = [annealer])
# Plot the loss and accuracy curves for training and validation 

plt.plot(history.history['val_loss'], color = 'b', label = "validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Final Accuracy

final_loss, final_acc = model.evaluate(xVal, yVal, verbose = 0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
# Predict the values from the validation dataset

Y_pred = model.predict(xVal)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(yVal,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
# predict results

results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results, name = "Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv", index = False)