import os

import cv2

import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D



from tensorflow import keras

from tensorflow.keras import Model, Input, optimizers



from PIL import Image



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))
dataset = pd.read_csv('../input/train.csv')





train_dir = os.listdir("../input/train/train")

test_dir = os.listdir("../input/test/test")



#train_dir, test_dir

dataset.head()
#retrieving the images and storing them in the arrays

data = []

labels = []



for i in train_dir:

    try:

    

        image = cv2.imread("../input/train/train/"+i)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((64 , 64))

        

        data.append(np.array(resize_img))

        

        labels.append(dataset[dataset['id'] == i]['has_cactus'].values[0])

        

        

    except AttributeError:

        print('Error')
cactus = np.array(data)

labels = np.array(labels)

cactus.shape,labels.shape
#Shuffle the data

cactus,labels = shuffle(cactus,labels)

cactus = cactus.astype("float32")/255

labels = tf.keras.utils.to_categorical(labels)

x_train,x_test,y_train,y_test = train_test_split(cactus,labels,test_size=0.2,random_state=0)

x_train.shape,x_test.shape,y_train.shape,y_test.shape




model = Sequential()



model.add(Conv2D(64, (3, 3), input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.2))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.2))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(32))



model.add(Dense(2))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.summary()
history = model.fit(x_train, y_train, batch_size=32, epochs=8, validation_split=0.15)
plt.plot(history.history['loss'],label='Loss')

plt.plot(history.history['val_loss'],label="Val Loss")

plt.legend()
accuracy  = model.evaluate(x_test,y_test)

print("Test Accuracy:-",accuracy)
#retrieving the images and storing them in the arrays

test_data = []



test_df = pd.read_csv('../input/sample_submission.csv')

images = test_df['id'].values



for i in images:

    try:

    

        image = cv2.imread("../input/test/test/"+i)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((64 , 64))

        

        test_data.append(np.array(resize_img))

                

        

    except AttributeError:

        print('Error')
test_cactus = np.array(test_data)

test_cactus = test_cactus.astype("float32")/255

test_cactus.shape
pred = model.predict(test_cactus)

test_df['has_cactus'] = pred

test_df.to_csv('aerial-cactus-submission.csv', index = False)
