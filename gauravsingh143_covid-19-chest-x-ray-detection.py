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
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
#Declaring the image size and the train, test path of the dataset
# Resizinig all the images to (224,224)
IMAGE_SIZE = [224,224]

train_path = '/kaggle/input/covid19-chest-xray-detection/covid_update/Train'
test_path = '/kaggle/input/covid19-chest-xray-detection/covid_update/Test'
resnet = ResNet50(input_shape = IMAGE_SIZE + [3], weights='imagenet', include_top=False)
#Now we dont have to train the existing weights we just have to train our last layer.
for layer in resnet.layers:
  layer.trainable = False
#By using the Glob function we will be able to know our output classes.
folder = glob('/kaggle/input/covid19-chest-xray-detection/covid_update/*')
folder
len(folder)
#Now the next we need to Flatten our ResNet model.
#What is Flattening of layer, why is it required?
###We need to convert our 2D features to 1D features. Flatting is required when we have to convert our layer to a fully connected layer.

x = Flatten()(resnet.output)
#Adding our last laye
prediction = Dense(2, activation='softmax')(x)
model = Model(inputs = resnet.inputs, outputs = prediction)
model.summary()
#Compiling our model
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Scaling all the images between 0 to 1

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False)

# Performing only scaling on the test dataset

test_datagen = ImageDataGenerator(rescale=1./255)
# flow_from_directory means i am applying all the train_datagen techniques to all the images
# We need to provide the same traget_size as initialized in the IMAGE_SIZE
# If you have more than two classes we should use class_mode = categorical
# But if we have just two classses we can use class_mode = binary

train_set = train_datagen.flow_from_directory(train_path,
                                              target_size=(224,224),
                                              batch_size=32,
                                              class_mode = 'categorical')
# Applying the same techniques on the test dataset

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')
EarlyStopping = EarlyStopping(monitor='val_loss', patience=5 , mode='min', verbose=1)
history = model.fit(train_set, validation_data=test_set, epochs=50, steps_per_epoch=len(train_set), validation_steps=len(test_set), callbacks=EarlyStopping)
#Plotting the losses and Accuracy on the dataset

plt.figure(figsize=(10,7))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title("Training Loss on COVID-19 Dataset")
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Training Accuracy on COVID-19 Dataset")
plt.legend()
plt.show()
plt.savefig('lossval_loss')
#Saving the h5 file
from tensorflow.keras.models import load_model

model.save('covid_chest_xray_model.h5')
pred = model.predict(test_set)
pred

import numpy as np

pred = np.argmax(pred, axis=0)
pred

#Loading our model¶

model = load_model('covid_chest_xray_model.h5')
from tensorflow.keras.preprocessing import image

img = image.load_img('/kaggle/input/covid19-chest-xray-detection/covid_update/Test/covid/covid-19-pneumonia-bilateral.jpg', target_size=(224,224))
x = image.img_to_array(img)
x
x.shape
x = x/255
from tensorflow.keras.applications.resnet50 import preprocess_input

x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape
model.predict(img_data)
a=np.argmax(model.predict(img_data), axis=1)
if(a==1):
    print("The person does not have COVID-19")
else:
    print("The person has COVID-19")
