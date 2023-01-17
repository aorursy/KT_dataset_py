#import the necessary libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D, MaxPooling2D, Flatten,Dropout

from tensorflow.keras.optimizers import Adam,SGD

#import the test and train dataset

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train= pd.read_csv("../input/digit-recognizer/train.csv")

df_test = pd.read_csv("../input/digit-recognizer/test.csv")
#Preprocessing of data-Check if there is any null values

print(df_train.isnull().sum())
print(df_test.isnull().sum())
df_train.shape
df_test.shape
#Split the train dataset into x_train of pixels and y_train of labels.Convert the data type to float so as to make computations easier

y_train = df_train["label"].astype('float32')

x_train = df_train.drop(labels = ["label"],axis = 1).astype('float32')
test = df_test.astype('float32')
#divide each pixel column of x_train by 255 as pixels range from(0,255)

x_train = x_train / 255.0
test = test / 255.0
#Reshaping the data for 2D CNN input into (-1,28,28,1) size

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print(x_train.shape)

print(test.shape)
# Some examples

g = plt.imshow(x_train[8][:,:,0])

print(y_train[8])
# Encode labels to one hot vectors 

y_train = keras.utils.to_categorical(y_train, 10)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=42)
model=Sequential()

#First Conv layer

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation ='relu', input_shape = (28,28,1)))

#Second Conv layer

model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))

#Applying Max Pooling to extract the max of the features obtained from the convolutions

model.add(MaxPooling2D(pool_size=(2,2)))



#Third Conv Layer

model.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))

#Applying Max Pooling to extract the max of the features obtained from the convolutions

model.add(MaxPooling2D(pool_size=(2,2)))

#Flattening the 3 dimensional feature map to 1 dimentional set of features

model.add(Flatten())

#Applying Fully connected layers

#Dense I layer

model.add(Dense(256, activation = "relu"))

#Applying Dropout to improve the correlation and tuning of the connections between layers(hyperparamter tuning)

model.add(Dropout(0.5))

#Output layers of 10 units as numbers range(0,9)

model.add(Dense(10, activation = "softmax"))
#Compiling the model with simple Adam optimizer and loss of categorical crossentropy and metrics involving of accuracy

model.compile(optimizer = Adam(learning_rate=0.001) , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
#Improving the accuracy of model by applying Data augmentation 

#Data augmentation is simply the different views of an image like 90* view,horizontal view,etc..

#It is basically used to improve the model and will be helpful if less train data is available

#By applying DA we get many augmented images of that same image which helps us in generalizing the data easily

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

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





datagen.fit(x_train)
#Fitting the model

epochs = 2

batch_size = 16

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val,y_val),

                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size)
#Predicting the test set results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
#Concatenating the required columns and creating a csv file

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("Predictions.csv",index=False)