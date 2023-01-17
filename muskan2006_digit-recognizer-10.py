# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')
#loading the dataset
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#viewing the image of the digit at index 35
%matplotlib inline
import matplotlib.pyplot as plt 
image_index=35
print(y_train[image_index])
plt.imshow(x_train[image_index],cmap='Greys')
plt.show()
#checking the shape of the training data
print(x_train.shape)
print(x_test.shape)
#checking the type of the data
type(x_train)
x_train.dtype
#checking for null values in training data
np.isnan(x_train).any()
#checking for null values in test data
np.isnan(x_test).any()

#checking for null values in training labels data
np.isnan(y_train).any()
#checking for null values in test labels data
np.isnan(y_test).any()
#reshaping 3-D array to 2-D array
x_train1=x_train.reshape(60000,784)
x_train1.shape
#converting the numpy array into pandas Datframe for further analysis
df=pd.DataFrame.from_records(x_train1)
df.shape
df.head()
df.describe()
#checking the index of a pixel with intensity 254 
df[df.iloc[:,774]==254].index
#checking the shape of the training labels data 
y_train.shape
#converting the training labels data into a dataframe
df_y=pd.DataFrame.from_records(y_train.reshape(60000,1))
df_y[0].head()
#displaying the count of each value (0,1,2,3,4,5,6,7,8,9) in the labels
df_y[0].value_counts()
#plotting a bar plot to show the frequency of each digit in the training data
df_y[0].value_counts().plot(kind='bar')
plt.show()
x_test1=x_test.reshape(10000,784)
df_2=pd.DataFrame.from_records(x_test1)
df_2.head()
#Preprocessing 
#Reshaping the images to a single color channel
x_train = df.values.reshape((-1, 28, 28, 1))
x_test = df_2.values.reshape((-1, 28, 28, 1))
#Checking the shape after re-shaping
print(x_train.shape)
print(x_test.shape)
#One-hot encoding on target
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes = 10)
y_train.shape
y_test = to_categorical(y_test, num_classes = 10)
y_test.shape
#preparing pixel data
#converting integers into float
train_norm_x = x_train.astype('float32')
test_norm_x = x_test.astype('float32')
#normalizing by dividing by the highest value i.e. 255
train_norm_x=train_norm_x/255.0
test_norm_x=test_norm_x/255.0
#splitting into training and validation data
X_train, X_val, Y_train, Y_val = train_test_split(train_norm_x, y_train, test_size = 0.1, random_state=2)
#making a basic cnn model of 2 convolutional layers without data augmentation
##model building
model_b = Sequential()
#convolutional layer with rectified linear unit activation
model_b.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
#32 convolution filters used each of size 3x3
#again
model_b.add(Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model_b.add(MaxPool2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model_b.add(Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model_b.add(Flatten())
#fully connected to get all relevant data
model_b.add(Dense(128, activation='relu'))
#one more dropout for convergence' sake :) 
model_b.add(Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model_b.add(Dense(10, activation='softmax'))
model_b.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
model_b.fit(X_train,Y_train, batch_size=32, epochs=10,verbose=1,validation_data = (X_val,Y_val))

#Data Augmentation 
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
datagen.fit(X_train)
model_b.fit_generator(datagen.flow(X_train,Y_train, batch_size=32),
                      epochs =10, validation_data = (X_val,Y_val),verbose = 2)