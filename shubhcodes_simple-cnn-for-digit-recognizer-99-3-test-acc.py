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
#Necessay Imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import adam

from keras.models import Model 

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D
train_df = pd.read_csv('../input/digit-recognizer/train.csv')

test_df = pd.read_csv('../input/digit-recognizer/test.csv')
x_train = train_df.drop(['label'], axis=1)

y_train = train_df['label']
sns.countplot(y_train)
#Performing One-hot-encoding

y_train = to_categorical(y_train, num_classes = 10)
# spliting the data into train and validation set with 9:1 ratio

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=.1, random_state=42)
#Reshaping the data for 2D CNN input into (-1,28,28,1) size

x_train = np.array(x_train).reshape(-1,28,28,1)

x_val = x_val.values.reshape(-1,28,28,1)

x_test = np.array(test_df).reshape(-1,28,28,1)
#Normalizing the image value by dividing train value by 255.0

x_train = x_train / 255.0

x_val = x_val / 255.0

x_test = x_test/255.0
#Printing the shapes 

print(x_train.shape)

print(x_val.shape)

print(x_test.shape)
#Using ImageDataGenerator for data augmentation

train_x_gen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.3, # Randomly zoom image 

        width_shift_range=0.3,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.3,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  





train_x_gen.fit(x_train)
#creating a model

model = Sequential([

                    Conv2D(64,(3,3), activation='relu', padding = 'same',input_shape=(28,28,1)),

                    Conv2D(64,(3,3), activation='relu', padding='same'),

                    MaxPooling2D(pool_size=(2,2)),

                    Dropout(.5),

                    Conv2D(64,(3,3), activation='relu', padding='same'),

                    Conv2D(64,(3,3), activation='relu', padding='same'),

                    MaxPooling2D(pool_size=(2,2)),

                    Dropout(.5),



                    Flatten(),

                    Dense(256, activation='relu'),

                    Dense(256, activation='relu'),

                    Dense(128, activation='relu'),

                    Dense(10, activation='softmax')



])
opt = adam(lr=.0001)

model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
#In this section we are going to train our model with fit_generator

# Fit the model

history = model.fit_generator(train_x_gen.flow(x_train,y_train, batch_size=128),

                              epochs = 10, validation_data = (x_val,y_val),

                              verbose = 2, steps_per_epoch=x_train.shape[0] // 128)
#testing the model of test.csv

predictions = model.predict(x_test)
#this code is going to build the loss and accuracy graph

plt.figure()

fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,7))

ax1.plot(history.history['loss'])

ax1.plot(history.history['val_loss'])

ax1.legend(['training','validation'])

ax1.set_title('loss')

ax1.set_xlabel('epoch')



ax2.plot(history.history['accuracy'])

ax2.plot(history.history['val_accuracy'])

ax2.legend(['training','validation'])

ax2.set_title('Acurracy')

ax2.set_xlabel('epoch')







score =model.evaluate(x_val,y_val,verbose=0)

print('Test Score:',score[0])

print('Test Accuracy:',score[1])
predictions.shape

pred_1 = model.predict_classes(x_test)
#this code is used to generate the csv for final submission

#uncomment the last three lines for saving csv

#sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

#sub['Label'] = pred_1

#sub.to_csv('../input/digit-recognizer/Submission.csv',index=False)