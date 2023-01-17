import numpy as np

from PIL import Image

import os

print(os.listdir("../input/cell_images/cell_images"))



# Any results you write to the current directory are saved as output.
import numpy as np

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from sklearn.metrics import precision_score, recall_score

from keras.layers import Dense,BatchNormalization,Activation,Dropout,Flatten,Conv2D,MaxPooling2D,AveragePooling2D
img_width, img_height = 64,64

l_rate = 0.001

_batch_size = 16

_epochs = 12

train_dir = "../input/cell_images/cell_images"

#model

model = Sequential()



model.add(Conv2D(16,(3,3), input_shape=(img_width, img_height, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,(3,3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3,3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(1024,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1024,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.summary()



SGD = optimizers.sgd(lr = l_rate, decay= 1e-6 , momentum=0.8, nesterov=True)

model.compile(

loss='binary_crossentropy',

optimizer=SGD,

metrics=['accuracy']

)

print('Compiled')

datagen = ImageDataGenerator(

rescale= 1./255,

horizontal_flip=True,

vertical_flip=True,    

validation_split=0.2

)

train_generator = datagen.flow_from_directory(

directory=train_dir,

target_size=(img_width, img_height),

classes=['Parasitized','Uninfected'],

class_mode='binary',

batch_size=_batch_size,

subset='training'

)

validation_generator = datagen.flow_from_directory(

directory=train_dir,

target_size=(img_width, img_height),

classes=['Parasitized','Uninfected'],

class_mode='binary',

batch_size=_batch_size,

subset='validation'

)

training = model.fit_generator(

generator=train_generator,

steps_per_epoch=1378,

epochs= _epochs,

validation_steps=344,

validation_data=validation_generator,

)

print('training done')

#model.save_weights('E://kaggle_malaria_detection//models//Malaria_cnn.h5')
import matplotlib.pyplot as plt

rand_img_P = "../input/cell_images/cell_images/Parasitized/C101P62ThinF_IMG_20150918_151149_cell_69.png"

img_P = Image.open(rand_img_P)

rand_img_U = "../input/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_150041_cell_127.png"

img_U = Image.open(rand_img_U)

f, axarr = plt.subplots(2)

axarr[0].imshow(img_P)

axarr[1].imshow(img_U)

print('''Gonna predict these images.

         1st is Parasitized

         2nd is Uninfected''')
print('Classes : ',train_generator.class_indices)

img_P = np.asarray(img_P.resize((64,64)))

img_P = img_P.reshape(1,64,64,3)

img_U = np.asarray(img_U.resize((64,64)))

img_U = img_U.reshape(1,64,64,3)



x1 = model.predict_classes(img_P)

x2 = model.predict_classes(img_U)



print("Prediction for Parasitized Image : ",x1)

print("Prediction for Uninfected Image : ",x2)