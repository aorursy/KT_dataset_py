import os

import cv2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix, accuracy_score

data_dir ='../input/leapgestrecog/leapGestRecog/'

main_dir = os.listdir(data_dir)

x=[]

y=[]



for folder in tqdm(main_dir):

    main_sub_dir = os.listdir(data_dir+folder)

    for subfolder in main_sub_dir:

        gesture = subfolder.split('_',1)[-1]

        images = os.listdir(data_dir+folder+'/'+subfolder+'/')

        for image in images:

            data = cv2.imread(data_dir+folder+'/'+subfolder+'/'+image,cv2.IMREAD_COLOR)

            data = cv2.resize(data,(150,150),interpolation = cv2.INTER_AREA)

            x.append(np.array(data))

            y.append(gesture)

           

            
fig, ax = plt.subplots(5,4,figsize=(30,30))

for i in range(5):

    for j in range(4):

        l = np.random.randint(0,len(x))

        ax[i,j].imshow(x[l])

        ax[i,j].set_title(y[l])
x = np.array(x)

x.shape


le = LabelEncoder()

y = le.fit_transform(y)

y=to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.80,random_state=21)

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.20, random_state=21)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same',activation ='relu', input_shape = (150,150,3)))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 



model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(10, activation = "softmax"))
callback =ModelCheckpoint('weights.hdf5',verbose=1,monitor='val_accuracy',save_best_only=True)
datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)

model.compile(optimizer='adam',metrics =['accuracy'],loss='categorical_crossentropy')
mymodel=model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),

                     epochs=20,validation_data=(x_val,y_val),callbacks=[callback])
plt.plot(mymodel.history['accuracy'],'*-')

plt.plot(mymodel.history['val_accuracy'],'*-')

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(mymodel.history['loss'],'*-')

plt.plot(mymodel.history['val_loss'],'*-')

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
pred = model.predict_classes(x_test)

pred
actual=[]

for i in range(len(y_test)):

    actual.append(np.argmax(y_test[i]))

print(confusion_matrix(actual,pred))

accuracy_score(actual,pred)