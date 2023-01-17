import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

import numpy as np

import matplotlib.pyplot as plt

import cv2

import os

Parasitized_cell=os.listdir('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/')



uninfected_cell=os.listdir('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/')



print("parasitized cell:",len(Parasitized_cell))

print("Uninfcted cell:",len(uninfected_cell))
for i in range(5):

    img=cv2.imread('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/'+Parasitized_cell[i])

    plt.imshow(img)

    plt.title("Parasitized")

    plt.show()
for i in range(5):

    img=cv2.imread('../input/cell-images-for-detecting-malaria/cell_images/Uninfected/'+uninfected_cell[i])

    plt.imshow(img)

    plt.title("Uninfected")

    plt.show()
width = 68

height = 68
datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
trainDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',

                                           target_size=(width,height),

                                           class_mode = 'binary',

                                           batch_size = 16,

                                           subset='training')
trainDatagen.class_indices

valDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',

                                           target_size=(width,height),

                                           class_mode = 'binary',

                                           batch_size = 16,

                                           subset='validation')

model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',input_shape=(width,height,3)))

model.add(MaxPool2D(2,2))

model.add(Dropout(0.2))





model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPool2D(2,2))

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
history=model.fit_generator(generator=trainDatagen,

                            steps_per_epoch=len(trainDatagen),

                            epochs=6,

                            validation_data=valDatagen ,

                            validation_steps=len(valDatagen )

                           )

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epochs')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
testimg_path="../input/cell-images-for-detecting-malaria/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_163.png"

img=image.load_img(testimg_path,target_size=(68,68))

plt.imshow(img)



x=image.img_to_array(img)

x=np.expand_dims(x,axis=0)

images=np.vstack([x])

val=model.predict(images)

if val==0:

    plt.title("Paracitized")

else:

    plt.title("Uninfected")