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
#Importing Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import os

from tensorflow.keras.preprocessing import image

from zipfile import ZipFile
#Importing Libraries for Deep Learning

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Test and train directory

train_dir = "../input/cat-and-dog/training_set/training_set"

test_dir = "../input/cat-and-dog/test_set/test_set"





train_dir_cat = train_dir + '/cats'

train_dir_dog = train_dir + '/dogs'

test_dir_cat = test_dir + '/cats'

test_dir_dog = test_dir + '/dogs'
print("Training Images for Cats : {}".format(len(os.listdir(train_dir_cat))))

print("Training Images for Dogs : {}".format(len(os.listdir(train_dir_dog))))

print('---')

print("Testing Images for Cats : {}".format(len(os.listdir(test_dir_cat))))

print("Testing Images for Dogs : {}".format(len(os.listdir(test_dir_dog))))
data_generator = ImageDataGenerator(rescale=1.0/255.0,zoom_range=0.2)
training_data = data_generator.flow_from_directory(directory = train_dir,

                                                  target_size= (64,64),

                                                  batch_size=32,

                                                  class_mode='binary')
testing_data = data_generator.flow_from_directory(directory=test_dir,

                                                 target_size = (64,64),

                                                 batch_size=32,

                                                 class_mode='binary')
#First define our sequential model

model = Sequential()
#First CNN Layer

model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=training_data.image_shape))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.3))
#Second CNN Layer

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))
#Third CNN Layer

model.add(Conv2D(filters=126,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.15))
#Fourth Flatten layer

model.add(Flatten())
#Fifth CNN Dense Layer

model.add(Dense(units=32,activation='relu'))

model.add(Dropout(rate=0.15))
#Sixth CNN Dense Layer

model.add(Dense(units=64,activation='relu'))

model.add(Dropout(rate=0.1))
#Final CNN Layer

model.add(Dense(units=len(set(training_data.classes)),activation='softmax'))
#Compile our layer

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#Lets see the summary of our model

model.summary()
#Fit the model

fit_model = model.fit_generator(training_data,

                               steps_per_epoch=1000,

                               epochs=25,

                               validation_data=testing_data,

                               validation_steps=1000                           

                               )
#Lets Visualize accuracy per steps 

accuracy = fit_model.history['accuracy']

plt.plot(range(len(accuracy)),accuracy,'bo',label='accuracy')

plt.legend()
#Lets test our model

def testing_image(image_dir):

    test_image = image.load_img(image_dir,target_size=(64,64))

    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image,axis=0)

    result = model.predict(x=test_image)

    print("Probability being Dog :{:.5f}".format(result[0][0]))

    print("Probability being Cat :{:.5f}".format(result[0][1]))

    if result[0][0] == 1:

        prediction = 'Hence, this is : Dog'

    else:

        prediction = 'Hence, this is : Cat'

        

    return prediction
print(testing_image(test_dir + '/cats/cat.4003.jpg'))