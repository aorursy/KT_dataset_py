# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen =  ImageDataGenerator(rescale=1/255,   rotation_range=30,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    # Your Code Here
    "/kaggle/input/cat-and-dog/training_set/training_set",
    target_size = (150,150),
    batch_size= 100,
    class_mode = 'binary'

)

test_datagen =  ImageDataGenerator(rescale=1/255)

test_generator = train_datagen.flow_from_directory(
    # Your Code Here
    "/kaggle/input/cat-and-dog/test_set/test_set",
    target_size = (150,150),
    batch_size= 100,
    class_mode = 'binary'

)
train_generator.class_indices

class myCallback(tf.keras.callbacks.Callback):
     # Your Code
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('acc')>0.9800:
            print("/nRequired accuracy of 99.9% reached so cancelling training")
            self.model.stop_training = True

callbacks = myCallback()
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32,(3,3),activation ="relu",input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(64,(3,3),activation ="relu"),
    tf.keras.layers.MaxPool2D(2,2),
    
    tf.keras.layers.Conv2D(128,(3,3),activation ="relu"),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.2),

    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation = tf.nn.relu),
    tf.keras.layers.Dense(128,activation = tf.nn.relu),
    tf.keras.layers.Dense(1,activation = "sigmoid"),

])
model.summary()
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer = Adam(lr=0.001), loss = 'binary_crossentropy',metrics = ['acc']
)

res = model.fit_generator(
    train_generator,
    steps_per_epoch=20, #batches to train per epoch
    epochs = 40,
    verbose=2,
    validation_data=test_generator,
    validation_steps=7

#     callbacks=[callbacks]

      # Your Code Here
)

# model fitting
res.history['acc'][-1]
model.save('lastnighttry1.h5')

from IPython.display import FileLink
FileLink(r'/kaggle/working/lastnighttry1.h5')
import cv2
# image = cv2.imread(path)
# a = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

path = "/kaggle/input/cat-and-dog/test_set/test_set/cats/cat.4721.jpg"

# /kaggle/input/cat-and-dog/test_set/test_set/cats/cat.4721.jpg
image = cv2.imread(path)
a = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
b= cv2.resize(a,(150,150))
c= b/255

import matplotlib.pyplot as plt
plt.imshow(c)
x = np.expand_dims(c, axis=0)
classw = model.predict(x)
print(classw)

import numpy as np
from keras.preprocessing import image


  # predicting images
path = [
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.191.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1154.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3843.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3889.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2942.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3154.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1728.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3890.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3467.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2167.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3918.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2403.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1720.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2242.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3813.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3949.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3323.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2111.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1121.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3412.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2686.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1589.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3272.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1555.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2889.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1248.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2962.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2800.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1487.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.832.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1480.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1172.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.594.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2668.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.199.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3258.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3774.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2579.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1790.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.379.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.220.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.750.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1380.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1292.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2831.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1819.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.597.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2855.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2199.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3966.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3431.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.441.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.494.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2233.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2510.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1604.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3278.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.72.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.406.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3508.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2806.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1117.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3269.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2028.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.344.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1686.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.846.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3494.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3221.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.598.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2434.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2177.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.763.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1566.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1685.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3024.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1479.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.840.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1807.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2993.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2521.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1207.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2041.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1894.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.65.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.3639.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1671.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.481.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.275.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2973.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.977.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.337.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.280.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1501.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.575.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.327.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.2068.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1852.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1947.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.215.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.384.jpg",
"/kaggle/input/cat-and-dog/training_set/training_set/cats/cat.1975.jpg",


]
total =0
sums=0
for i in path:
    img = image.load_img(i, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    total+=1
    if classes[0]>0.5:
        print( " is a dog")

    else:
        print(" is a cat")
        sums+=1


print("test accuracy is ",(sums/total)*100)

