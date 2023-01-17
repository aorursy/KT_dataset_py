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
import keras

import numpy as np

from glob import glob

import matplotlib.pyplot as plt

from keras.applications import VGG19

from keras.models import Sequential,Model

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator,img_to_array

from keras.layers import Flatten,Dense,Dropout
img_size = [224,224]

vgg_model = VGG19(include_top = False,weights = 'imagenet',input_shape = img_size+[3])

for i in vgg_model.layers:

    i.trainable = False
x = Flatten()(vgg_model.output)

x = Dense(1024,activation = 'relu',kernel_initializer='he_uniform')(x)

x = Dropout(0.5)(x)

x = Dense(512,activation = 'relu',kernel_initializer='he_uniform')(x)

prediction = Dense(1,activation = 'sigmoid')(x)

model = Model(inputs = vgg_model.input,outputs = prediction)

model.summary()
data_gen = ImageDataGenerator(rescale= 1.0/255.0,validation_split=0.3)



train =data_gen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images',

                                    target_size = (224,224),class_mode = 'binary',

                                    batch_size = 64,subset = 'training'

                                    )



validation = data_gen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images',

                                          target_size = (224,224),class_mode = 'binary',

                                    batch_size = 64,subset = 'validation')



classes = ['Parasitized', 'Uninfected']
model.compile('adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

es = EarlyStopping(monitor='val_loss',patience=3)

data = model.fit(train,epochs = 20,callbacks = [es],validation_data = validation)
ypred = model.predict(validation,verbose = 1)

performance =model.evaluate_generator(validation)

print(f"loss: {performance[0]}")

print(f"accuracy: {performance[1]}")
img = keras.preprocessing.image.load_img('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png',target_size=(224,224))

img = img_to_array(img)

img =  img/255.0

img = np.array([img])

(model.predict(img)>0.5).astype(int)
img = keras.preprocessing.image.load_img('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144348_cell_108.png',target_size=(224,224))

img = img_to_array(img)

img =  img/255.0

img = np.array([img])

(model.predict(img)>0.5).astype(int)
loss_train = data.history['loss']

loss_val = data.history['val_loss']

epochs = range(1,12)

plt.plot(epochs,loss_train,'g',label = 'training loss')

plt.plot(epochs,loss_val,'b',label = 'validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
loss_train = data.history['accuracy']

loss_val = data.history['val_accuracy']

epochs = range(1,12)

plt.plot(epochs,loss_train,'g',label = 'training accuracy')

plt.plot(epochs,loss_val,'b',label = 'validation accuracy')

plt.title('Training and Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()