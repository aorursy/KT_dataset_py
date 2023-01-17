# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
# Display the content of data

train.info()
# shape gives number of rows and columns in a tuple

train.shape
train.Id.describe()
train.head(10)
train.tail(10)
# put labels into y_train variable

y_train = train["Id"]

# Drop 'Id' column

X_train = train.drop(labels = ["Id"], axis = 1)

y_train.head()
# Indicates sum of values in our data

train.isnull().sum().sum()
import cv2

import glob

import numpy as np

import os, sys, time

import numpy as numpy





def filterForTails (img):

    draw = False

    waitValue = 10



    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(imgray, (7, 7), 0)

    ret,thresh = cv2.threshold(blur,127,255,0)



    blur2 = cv2.GaussianBlur(thresh, (7, 7), 0)

    thresh2 = cv2.threshold(blur2, 250, 250, cv2.THRESH_BINARY)[1]

    #blur3 = cv2.GaussianBlur(thresh2, (7, 7), 0)



    contours2, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    for c in contours2:



        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)



        x, y, w, h = cv2.boundingRect(c)

        roi = img[y:h+y, x:w+x]

        imgray2 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)



        if (len(approx) != 1):

            draw = True



    if (draw):

        img = cv2.drawContours(img, contours2, -3, (255, 255, 255), -8)



        draw = False

    return img
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input



def prepare_images(data, m, dataset, filterOn):

    print("Preparing images")

    X_train = np.zeros((m, 100, 100, 3))

    count = 0

    

    for fig in data['Image']:

       #load images into images of size 100x100x3

        img = cv2.imread("../input/"+dataset+"/"+fig)

        if(filterOn):

            img = filterForTails(img)

        img = cv2.resize(img, (100, 100))

        

        x = image.img_to_array(img)

        x = preprocess_input(x)

        

        X_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return X_train
filterOn = True



start = time.time()

x_train = prepare_images(train, train.shape[0], "train", filterOn)

end = time.time()

elapsed = end - start

print("Elapsed time: ", elapsed)
x_train = x_train / 255.0

print("x_train shape: ",x_train.shape)
# Some examples(first one)

plt.imshow(x_train[0][:,:,0], cmap="gray")

plt.title(plt.title(train.iloc[0,0]))

plt.axis("off")

plt.show()
# Some examples(last one)

plt.imshow(x_train[25360][:,:,0], cmap="gray")

plt.title(plt.title(train.iloc[25360,0]))

plt.axis("off")

plt.show()
# Some examples(55th)

plt.imshow(x_train[55][:,:,0], cmap="gray")

plt.title(plt.title(train.iloc[55,0]))

plt.axis("off")

plt.show()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
#let's look at first 10 values

y_train[0:10]  # => new_whale :)
y_train.shape
# convert to one-hot-encoding(one hot vectors)

# we have 5005 class look at from=> train.Id.describe()



from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 5005)
#converted

print(y_train.shape)

y_train #let's look at vectors
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential # to create a cnn model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (100,100,3)))

model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))

model.add(Dropout(0.25))



# fully connected

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dense(y_train.shape[1], activation = "softmax"))
model.summary()
# Define the optimizer

optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
# # Define the optimizer

# optimizer = RMSprop(lr = 0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
# if you want to use Data Augmentation,Activate it.
# # With data augmentation to prevent overfitting



# datagen = ImageDataGenerator(

#         featurewise_center=False,  # set input mean to 0 over the dataset

#         samplewise_center=False,  # set each sample mean to 0

#         featurewise_std_normalization=False,  # divide inputs by std of the dataset

#         samplewise_std_normalization=False,  # divide each input by its std

#         zca_whitening=False,  # apply ZCA whitening

#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

#         zoom_range = 0.1, # Randomly zoom image 

#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

#         horizontal_flip=False,  # randomly flip images

#         vertical_flip=False)  # randomly flip images





# datagen.fit(x_train)
epochs = 30  # for better result increase the epochs

batch_size = 1000
#if you don't want to use data augmentation ,Use this code.

start = time.time()

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[learning_rate_reduction])

end = time.time()

elapsed = end - start
print("Elapsed time: ", elapsed)
# history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),

#                               epochs=100, verbose = 2, 

#                               steps_per_epoch=x_train.shape[0] // batch_size,

#                               callbacks=[learning_rate_reduction]) 
# Plot the loss curve for training

plt.plot(history.history['loss'], color='r', label="Train Loss")

plt.title("Train Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# Plot the accuracy curve for training

plt.plot(history.history['accuracy'], color='g', label="Train Accuracy")

plt.title("Train Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
print('Train accuracy of the model: ',history.history['accuracy'][-1])
print('Train loss of the model: ',history.history['loss'][-1])
test = os.listdir("../input/test/")

print(len(test))
col = ['Image']

test_data = pd.DataFrame(test, columns=col)

test_data['Id'] = ''
x_test = prepareImages(test_data, test_data.shape[0], "test")

x_test /= 255
predictions = model.predict(np.array(x_test), verbose=1)
for i, pred in enumerate(predictions):

    test_data.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_data.head(10)

test_data.to_csv('submission_3.csv', index=False)