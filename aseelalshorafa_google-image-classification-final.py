# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
img_dir = '../input/image-classification/images/images'

val_dir = '../input/image-classification/validation/validation'

test_dir = '../input/image-classification/test/test'
#  ! pip install -q tf-nightly
import tensorflow as tf

from tensorflow import keras

tf.__version__
# change as you want

image_size = 256

batch_size = 32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="training",

    seed=1007,

    image_size=(image_size,image_size),

    batch_size=batch_size,

    label_mode='categorical'



)
class_names = train_ds.class_names
val_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=img_dir,

    validation_split=0.2,

    subset="validation",

    seed=1007,

    image_size=(image_size,image_size),

    batch_size=batch_size,

    label_mode='categorical'

)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(

    directory=test_dir,

    image_size=(image_size,image_size),

    batch_size=batch_size,

    label_mode='categorical',



)
# put your code here 

import matplotlib.pyplot as plt

# x_shape = []

# y_shape = []

plt.figure(figsize=(12,15))

for img,label in train_ds.take(1):

    for i in range(16):

#         x_shape.append(img[i].shape[0])

#         y_shape.append(img[i].shape[1])

        ax = plt.subplot(4,4,i+1)

        plt.imshow(img[i].numpy().astype("uint8"))

        plt.title(class_names[np.argmax(label[i])])

       

    
    #resnet152 96

    # DenseNet201 97

    # Mobilenetv2 96

    from tensorflow.keras.applications.densenet import DenseNet201,preprocess_input
# put your code here 

base_model = DenseNet201(weights = 'imagenet',

                   include_top = False,

                  input_shape=(image_size,image_size,3))

base_model.trainable = False
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten,AveragePooling2D,Dense,Lambda,Conv2D,BatchNormalization,AveragePooling1D

model = Sequential()

model.add(Lambda(preprocess_input,input_shape = (image_size,image_size,3)))

model.add(base_model)

model.add(BatchNormalization())

model.add(AveragePooling2D())

model.add(Flatten())

model.add(Dense(128,activation = 'relu'))

model.add(Dense(64,activation = 'relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dense(4,activation = 'softmax'))



model.summary()
model.compile(optimizer='adam',loss = 'CategoricalCrossentropy',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping 

s = EarlyStopping(patience = 3, monitor = 'val_accuracy')
model.fit(train_ds,validation_data=val_ds,epochs = 50 , batch_size=32,callbacks=[s])
# put your code here 

history = pd.DataFrame(model.history.history)

history.plot()

# put your code here 

model.save('my_model.h5')
prediction = model.predict_classes(test_ds)
prediction
predictions = []

for i in prediction:

    predictions.append(class_names[i])

    
plt.figure(figsize=(12,15))

for image,label  in test_ds.take(1):

        for i in range(9):

            plt.subplot(3,3,i+1)

            plt.imshow(image[i].numpy().astype("uint8"))

            plt.title(predictions[i])

from tensorflow.keras.preprocessing.image import load_img,img_to_array

from tensorflow.keras.utils import get_file

images = [

   'https://images.freeimages.com/images/large-previews/575/food-1489046.jpg',

    'https://images.freeimages.com/images/large-previews/3a7/travel-sweden-norway-47-1471775.jpg',

    'https://images.freeimages.com/images/large-previews/7a3/architectural-details-12-1234752.jpg'

]

files_path = []

counter = 0

for i in images:

        file = get_file(origin=i ,fname='picture{}.jpg'.format(counter+1),cache_subdir='datasets/images')

        counter = counter+1

        files_path.append(file)

        
files_path
test_ds_test = tf.keras.preprocessing.image_dataset_from_directory(

    directory='/root/.keras/datasets',

    image_size=(image_size,image_size),

    batch_size=batch_size,

    label_mode='categorical',



)
prediction = model.predict_classes(test_ds_test)

predictions = []

for i in prediction:

    predictions.append(class_names[i])

plt.figure(figsize=(12,15))

for s,label  in test_ds_test.take(1):

        for i in range(3):

            plt.subplot(1,3,i+1)

            plt.imshow(s[i].numpy().astype("uint8"))

            plt.title(predictions[i])