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
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train_data = train.loc[:, "pixel0":]
train_label= train.loc[:, "label"]

train_data = np.array(train_data)

train_label = tf.keras.utils.to_categorical(train_label, num_classes=10, dtype='float32')


test_data = test.loc[:, "pixel0":]
test_data = np.array(test_data)


train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data  = test_data.reshape(test_data.shape[0],28,28,1)

train_data = train_data/255.0
test_data  = test_data/255.0

train_data= tf.keras.layers.ZeroPadding2D(padding=50)(train_data)
test_data=  tf.keras.layers.ZeroPadding2D(padding=50)(test_data)

#I needed to pad the images since the inception model can't accept images with dimensions less than (75*75)




# train_datagen = ImageDataGenerator.flow_from_directory(directory="/kaggle/input/digit-recognizer/train.csv",
#     target_size=(150, 150),
#     color_mode="grey",
#     classes=10,
#     class_mode="categorical",
#     shuffle=True,
#     save_format="png",
#     interpolation="nearest",
# )
print(train_data.shape)
print(test_data.shape)
from tensorflow.keras import layers,Input
from tensorflow.keras import Model
from tensorflow import keras
from keras.preprocessing.image import *
from matplotlib import pyplot
import os

   
pre_trained_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights=None,
    input_tensor=Input(shape=(128,128, 1))
)


#Freezing the layers in order to prevent the framework from training the base model
for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1024, activation= 'elu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (10, activation='softmax')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
# train_datagen = ImageDataGenerator(
#                                    rotation_range = 40,
#                                    width_shift_range = 0.2,
#                                    height_shift_range = 0.2,
#                                    shear_range = 0.2,
#                                    zoom_range = 0.2,
#                                    horizontal_flip = True)


history = model.fit(train_data,train_label,epochs=200,batch_size = 32)
model.save('mymodel.h5')
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.legend()
plt.figure()

loss = history.history['loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()
history = model.fit(train_data,train_label,epochs=20,batch_size = 64)
predictions = model.predict(test_data)

prediction = []

for i in predictions:
    prediction.append(np.argmax(i))

    
submission =  pd.DataFrame({
        "ImageId": test.index+1,
        "Label": prediction
    })

submission.to_csv('submission5.csv', index=False)
