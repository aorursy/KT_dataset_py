# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Sequential,Model

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,Activation,BatchNormalization

from keras import losses

from keras.optimizers import Adam, Adagrad

from keras.callbacks import EarlyStopping

from keras import regularizers

from keras.models import model_from_json

from sklearn.model_selection import GridSearchCV

import keras

from keras.layers import LeakyReLU

import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
model = Sequential()

model.add(Conv2D(16,kernel_size = (11,11),input_shape=(224,224,3), padding="valid",

                 activity_regularizer = regularizers.l2(1e-8)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(4, 4),strides=2, data_format=None))

model.add(Dropout(0.1))



model.add(Conv2D(32,kernel_size = (5,5), padding="valid"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(4, 4),strides=2, data_format=None))





model.add(Conv2D(64,kernel_size = (5,5),  padding="valid"))

model.add(Activation("relu"))

model.add(Conv2D(64,kernel_size = (3,3), padding="valid", activity_regularizer = regularizers.l2(1e-8)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(4, 4),strides=2, data_format=None))



model.add(Conv2D(128,kernel_size = (3,3),  padding="valid"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(4, 4),strides=2, data_format=None))

model.add(Conv2D(128,kernel_size = (2,2), padding="valid"))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2),strides=2, data_format=None))

model.add(Dropout(0.1))



model.add(Flatten())



model.add(Dense(512, activity_regularizer = regularizers.l2(1e-8)))

model.add(Activation("relu"))

model.add(Dropout(0.1))



model.add(Dense(256))

model.add(Activation("relu"))

model.add(Dropout(0.1))





model.add(Dense(64))

model.add(Activation("relu"))





model.add(Dense(5,activation = 'softmax'))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001, 

                                                    beta_1=0.9, beta_2=0.999, epsilon=None, 

                                                    decay=0.0, amsgrad=False), metrics=["accuracy"])

model.summary()
import os

input_path = os.path.join(os.getcwd(),"../input/train_resizedvgg net")

test_path = os.path.join(os.getcwd(),"../input/train_resizedvgg net")

print(os.listdir(test_path))

print(input_path)
from keras.preprocessing.image import ImageDataGenerator

import PIL



train_datagen = ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

test_datagen = ImageDataGenerator()





train_generator = train_datagen.flow_from_directory( input_path,

   # '../input/train_resizedvgg net/',

    target_size=(224,224),

    batch_size=32

)

validation_generator = test_datagen.flow_from_directory( test_path,

     #   '../input/test_resizedvgg net/',

        target_size=(224,224),

        batch_size=32)



modelhist = model.fit_generator(

        train_generator,

        steps_per_epoch=len(train_generator),

        epochs=10,

        validation_data=validation_generator,

        validation_steps=50

        )
model_name = 'model.h5'

save_dir = os.getcwd()

# Save model and weights

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)



# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
import matplotlib.pyplot as plt

plt.plot(modelhist.history['loss'])

plt.plot(modelhist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

plt.savefig("km_loss.png")

plt.close()
plt.plot(modelhist.history['acc'])

plt.plot(modelhist.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('acc')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.savefig("km_acc.png")

plt.show()

plt.close()
loss, acc = model.evaluate_generator(validation_generator,verbose=2,

                                           steps=len(validation_generator))

print("Restored model, accuracy: {:5.2f}%".format(100*acc))
loaded_model = keras.models.load_model(model_path)

loaded_model.summary()
loss, acc = loaded_model.evaluate_generator(validation_generator,verbose=2,

                                           steps=len(validation_generator))

print("Restored model, accuracy: {:5.2f}%".format(100*acc))