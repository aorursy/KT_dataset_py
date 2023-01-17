# Import 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.image import imread

import cv2

import seaborn as sns

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report,confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image

plt.rcParams['figure.figsize'] = (12,7)



# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images"))
infected = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized")

infected_path = "../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized"

print("Length of infected data = ",len(infected),'images')

uninfected = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected")

uninfected_path = "../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected"

print("Length of uninfected data = ",len(uninfected),'images')
for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(cv2.imread(infected_path+'/'+infected[i]))

    plt.title('PARASITIZED CELL')

    plt.tight_layout()

plt.show()
for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(cv2.imread(uninfected_path+'/'+uninfected[i]))

    plt.title('UNINFECTED CELL')

    plt.tight_layout()

plt.show()
dim1 = []

dim2 = []

for file in infected:

    try:

        imag = imread(infected_path+'/'+file)

        d1,d2,colors = imag.shape

        dim1.append(d1)

        dim2.append(d2)

    except:

        None
sns.jointplot(dim1,dim2)
print('Mean of X dimensions - ',np.mean(dim1))

print('Mean of Y dimensions - ',np.mean(dim2))
cv2.imread(infected_path+'/'+infected[0]).max()
img_shape = (130,130,3)

image_gen = ImageDataGenerator(rotation_range = 20,

                              width_shift_range = 0.1,

                              height_shift_range=0.1,

                              rescale=1 / 255,

                              shear_range=0.1,

                              zoom_range=0.1,

                              horizontal_flip=True,

                              fill_mode='nearest',

                              validation_split=0.2)
image_gen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images')
train = image_gen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images',

                                     target_size =img_shape[:2],

                                     color_mode='rgb',

                                     batch_size = 16,

                                     class_mode='binary',shuffle=True,

                                     subset="training")



validation = image_gen.flow_from_directory('../input/cell-images-for-detecting-malaria/cell_images/cell_images',

                                     target_size = img_shape[:2],

                                     color_mode='rgb',

                                     batch_size = 16,

                                     class_mode='binary',

                                     subset="validation",shuffle=False)



train.class_indices
# Model 1 ---

model = Sequential()



model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape = (130,130,3),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
early = EarlyStopping(monitor='val_loss',patience=2,verbose=1)
model.metrics_names
model.fit_generator(train,

                   epochs=20,

                   validation_data=validation,

                   callbacks=[early])
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
losses[['accuracy','val_accuracy']].plot()
predictions = model.predict_generator(validation)
predictions = predictions>0.5 # The most important factor, directly control precision and recall.
print('Confusion Matrix: \n',confusion_matrix(validation.classes,predictions),'\n')

print('Classification Report: \n\n',classification_report(validation.classes,predictions))
model.save('model.h5')
img = image.load_img(infected_path+'/'+infected[22],target_size = img_shape)

img
img_arr = image.img_to_array(img)
model.predict_classes(img_arr.reshape(1,130,130,3))