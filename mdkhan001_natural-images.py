

import numpy as np

import pandas as pd

import math

import os



import warnings

warnings.filterwarnings("ignore")



from keras.preprocessing.image import img_to_array

import random

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Convolution2D,BatchNormalization

from keras.layers import MaxPooling2D,Dropout

from keras.layers import Flatten

from keras.layers import Dense

import cv2

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split



print(os.listdir("../input/data/natural_images"))


data=[]

labels=[]

random.seed(42)

imagePaths = sorted(list(os.listdir("../input/data/natural_images")))

random.shuffle(imagePaths)

print(imagePaths)



for img in imagePaths:

    path=sorted(list(os.listdir("../input/data/natural_images/"+img)))

    for i in path:

        image = cv2.imread("../input/data/natural_images/"+img+'/'+i)

        image = cv2.resize(image, (128,128), interpolation=cv2.INTER_AREA)

        image = img_to_array(image)

        data.append(image)

        l = label = img

        labels.append(l)

        
print(len(data), len(labels))
# data[0]


data = np.array(data, dtype="float") / 255.0

labels = np.array(labels)

mlb = LabelBinarizer()

labels = mlb.fit_transform(labels)

print(labels[0])



(x_train,x_test,y_train,y_test)=train_test_split(data,labels,test_size=0.2,random_state=42)

print(len(x_train), len(x_test))



# Initialising the CNN

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Convolution2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu',padding='same'))

# Adding a second convolutional layer

classifier.add(Convolution2D(32, (3, 3), activation = 'relu',padding='same'))

# step 2 - Pooling layer

#classifier.add(BatchNormalization(axis=1))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))





classifier.add(Convolution2D(64, (3, 3), activation = 'relu',padding='same'))

classifier.add(Convolution2D(64, (3, 3), activation = 'relu',padding='same'))

#classifier.add(BatchNormalization(axis=1))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))





classifier.add(Convolution2D(128, (3, 3), activation = 'relu',padding='same'))

classifier.add(Convolution2D(128, (3, 3), activation = 'relu',padding='same'))

#classifier.add(BatchNormalization(axis=1))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))





classifier.add(Convolution2D(256, (3, 3), activation = 'relu',padding='same'))

classifier.add(Convolution2D(256, (3, 3), activation = 'relu',padding='same'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))



# Step 3 - Flattening

classifier.add(Flatten())

classifier.add(Dense(1024,activation='relu'))

classifier.add(BatchNormalization())

classifier.add(Dropout(0.5))

"""

classifier.add(Dense(512,activation='relu'))

classifier.add(BatchNormalization())

classifier.add(Dropout(0.5))

"""

# Step 4 - Full connection

classifier.add(Dense(output_dim = 8, activation = 'softmax'))



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



classifier.summary()





### 94% acc





classifier.fit(x_train,y_train, batch_size = 100, epochs = 10, validation_split=0.1)



classifier.save('CNN_natural-images_ep10.h5')


pred_label = classifier.predict(x_test, batch_size=None, verbose=0)



total = 0

accurate = 0

accurate_index = []

wrong_index = []



for i in range(len(pred_label)):

    if np.argmax(pred_label[i]) == np.argmax(y_test[i]):

        accurate += 1

        accurate_index.append(i)

    else:

        wrong_index.append(i)

        

    total += 1

    

    

print('Total test data;', total, '\taccurately predicted data:', accurate, '\t wrongly predicted data: ', total - accurate)

print('Accuracy:', round(accurate/total*100, 3), '%')



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip=True, shear_range=0.2)



batch_size = 100

classifier.fit_generator(train_datagen.flow(x_train,y_train,batch_size=batch_size), steps_per_epoch=math.ceil(len(x_train)//100), epochs = 10)



classifier.save('CNN_natural-images-with-augmentation_ep10.h5')


prediction_label = classifier.predict(x_test, batch_size=None, verbose=0)



total = 0

accurate = 0

accurate_index = []

wrong_index = []



for i in range(len(prediction_label)):

    if np.argmax(prediction_label[i]) == np.argmax(y_test[i]):

        accurate += 1

        accurate_index.append(i)

    else:

        wrong_index.append(i)

        

    total += 1

    

    

print('Total test data;', total, '\taccurately predicted data:', accurate, '\t wrongly predicted data: ', total - accurate)

print('Accuracy:', round(accurate/total*100, 3), '%')



labelss=['airplane','car','cat','dog','flower','fruit','motorbike','person']

im_idx = random.sample(accurate_index, k=9)



nrows = 3

ncols = 3

fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))



n = 0

for row in range(nrows):

    for col in range(ncols):

            ax[row,col].imshow(x_test[im_idx[n]])

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(labelss[np.argmax(prediction_label[im_idx[n]])], labelss[np.argmax(y_test[im_idx[n]])]))

            n += 1



plt.show()



im_idx = random.sample(wrong_index, k=9)



nrows = 3

ncols = 3

fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(15, 12))



n = 0

for row in range(nrows):

    for col in range(ncols):

            ax[row,col].imshow(x_test[im_idx[n]])

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(labelss[np.argmax(prediction_label[im_idx[n]])], labelss[np.argmax(y_test[im_idx[n]])]))

            n += 1



plt.show()
