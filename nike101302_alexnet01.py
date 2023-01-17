import os

import cv2

import os

import cv2

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.layers.normalization import BatchNormalization





categories = os.listdir('/kaggle/input/stanford-dogs-dataset/annotations/Annotation')

categories = categories[:3]

filepath='/kaggle/input/stanford-dogs-dataset/images/Images'
len(categories)
img_lst=[]

labels=[]



for index, category in enumerate(categories):

    for image_name in os.listdir(filepath+"/"+category):

        img = cv2.imread(filepath+"/"+category+"/"+image_name)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   

        img_array = Image.fromarray(img, 'RGB')            

        #resize image to 227 x 227 because the input image resolution for AlexNet is 227 x 227

        resized_img = img_array.resize((227, 227))

        img_lst.append(np.array(resized_img))



        labels.append(index)

    
len(labels)
images = np.array(img_lst)

labels = np.array(labels)

images = images.astype(np.float32)

labels = labels.astype(np.int32)

images = images/255
labels[710]
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.15, random_state = 34)



print("x_train shape = ",x_train.shape)

print("y_train shape = ",y_train.shape)

print("\nx_test shape = ",x_test.shape)

print("y_test shape = ",y_test.shape)
def display_rand_images(images, labels):

    plt.figure(1 , figsize = (19 , 10))

    n = 0 

    for i in range(9):

        n += 1 

        r = np.random.randint(0 , images.shape[0] , 1)

        

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)

        plt.imshow(images[r[0]])

        

        plt.title('Dog breed : {}'.format(labels[r[0]]))

        plt.xticks([])

        plt.yticks([])

        

    plt.show()

    

display_rand_images(images, labels)
from keras.layers import Dense

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.models import Sequential

from keras.layers import ZeroPadding2D

classifier = Sequential()

classifier.add(Convolution2D(96,(11,11), input_shape=(227,227,3),strides=(4,4), padding='valid', activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

classifier.add(BatchNormalization())

classifier.add(Convolution2D(256,(5,5),strides=(1,1), padding='valid', activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

classifier.add(BatchNormalization())

#classifier.add(ZeroPadding2D(padding=(1, 1)))

classifier.add(Convolution2D(384,(3,3),strides=(1,1), padding='valid', activation='relu'))

classifier.add(BatchNormalization())

#classifier.add(ZeroPadding2D(padding=(1, 1)))

classifier.add(Convolution2D(384,(3,3),strides=(1,1), padding='valid', activation='relu'))

classifier.add(BatchNormalization())

#classifier.add(ZeroPadding2D(padding=(1, 1)))

classifier.add(Convolution2D(256,(3,3),strides=(1,1), padding='valid', activation='relu'))

classifier.add(BatchNormalization())

#classifier.add(ZeroPadding2D(padding=(1, 1)))

classifier.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

classifier.add(BatchNormalization())

classifier.add(Flatten())

classifier.add(Dense(4096, input_shape=(224*224*3,),activation='relu'))

classifier.add(Dropout(0.4))

classifier.add(BatchNormalization())

classifier.add(Dense(4096 ,activation='relu'))

classifier.add(Dropout(0.4))

classifier.add(BatchNormalization())

classifier.add(Dense(1000 ,activation='relu'))

classifier.add(Dropout(0.4))

classifier.add(BatchNormalization())

classifier.add(Dense(3 ,activation='softmax'))

classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

classifier.summary()

classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
%%time

alexnetmodel=classifier.fit(x_train, y_train,validation_split=0.3, epochs=100)
import matplotlib.pyplot as plt

plt.plot(alexnetmodel.history['accuracy'])

plt.plot(alexnetmodel.history['loss'])

plt.plot(alexnetmodel.history['val_accuracy'])

#plt.title('model accuracy')

#plt.ylabel('accuracy')

#plt.xlabel('epoch')

#plt.legend(['train', 'test'], loc='upper left')

plt.show()
loss, accuracy = classifier.evaluate(x_test, y_test)



print(loss,accuracy)
pred = classifier.predict_classes(x_test)



pred.shape
randclass= np.random.randint(1,83,1)

randclass=int(randclass)



plt.imshow(x_test[randclass])

print('actual class '+ str(y_test[randclass]))

print(pred[randclass])
from sklearn import metrics

print(metrics.confusion_matrix(y_test, pred))