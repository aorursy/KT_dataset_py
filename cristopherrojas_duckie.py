import cv2

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



import os

import random

import gc
train_dir_duck='../input/duckietown/duck'

train_dir_road='../input/duckietown/road'

train_dir_two='../input/duckietown/two'

train_duck =['../input/duckietown/duck/{}'.format(i) for i in os.listdir(train_dir_duck) if 'duck' in i]

train_two =['../input/duckietown/two/{}'.format(i) for i in os.listdir(train_dir_two) if 'two' in i]

train_road =['../input/duckietown/road/{}'.format(i) for i in os.listdir(train_dir_road) if 'road' in i]



random.shuffle(train_duck)
print(len(train_road),

len(train_duck),len(train_two))
train_imgs = train_duck[:1400] + train_road[:2400] + train_two[:950]

test_imgs = train_duck[1400:1460] + train_road[2400:2524]+ train_two[950:987]



random.shuffle(train_imgs)
import matplotlib.image as mpimg

for ima in train_imgs[0:3]:

    img=mpimg.imread(ima)

    imgplot = plt.imshow(img)

    plt.show()
nrows= 150

ncolumns =150

channels = 3
def read_and_process_image(list_of_images):

    X=[]

    y=[]

    

    for image in list_of_images:

        try:

            X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncolumns),interpolation=cv2.INTER_CUBIC))

        

            if 'duck' in image[16:] or 'two' in image[16:]:

                y.append(1)

            elif 'road' in image[16:]:

                y.append(0)

        except Exception as e:

            print(str(e))

            

    return X, y
X, y = read_and_process_image(train_imgs)
np.unique(y,return_counts=True)
plt.figure(figsize=(20,10))

columns = 5

for i in range (columns):

    plt.subplot(5/columns+1,columns,i+1)

    plt.imshow(X[i])
import seaborn as sns

del train_imgs

gc.collect()



X=np.array(X)

y=np.array(y)



sns.countplot(y)

plt.title('labers for road and ducks')
print('shape of train images is', X.shape)

print('shape of labels is',y.shape)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train,y_val = train_test_split(X,y, test_size=0.20,random_state=2)



print("Shape of train images is", X_train.shape)

print("Shape of validation images is", X_val.shape)

print("Shape of labels is",y_train.shape)

print("Shape of labels is", y_val.shape)
del X

del y

gc.collect()



ntrain= len(X_train)

nval=len(X_val)



batch_size=32
#from keras.applications import InceptionResNetV2

#conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))

#conv_base.summary()
from keras import layers

from keras import models

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img
#model=models.Sequential()

#model.add(conv_base)

#model.add(layers.Flatten())

#model.add(layers.Dense(256,activation='relu'))

#model.add(layers.Dense(1,activation='sigmoid'))

#model.summary()
#print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))

#conv_base.trainable=False

#print('Number of trainable weights after freezing the conv base:', len(model.trainable_weights))
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-4),metrics=['acc'])
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

val_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow(X_train,y_train,batch_size=batch_size)

val_generator=val_datagen.flow(X_val,y_val,batch_size=batch_size)
history=model.fit_generator(train_generator,steps_per_epoch=ntrain//batch_size,epochs=90,validation_data=val_generator,validation_steps=nval//batch_size)
model.save_weights('model_weights.h5')

model.save('model_keras.h5')
acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(1,len(acc)+1)



plt.plot(epochs,acc,'b',label='Training accuaricy')

plt.plot(epochs, val_acc,'r',label='Validation accuaricy')

plt.title('Training and Validation accuaricy')

plt.legend()



plt.figure()



plt.plot(epochs,loss,'b', label='Training loss')

plt.plot(epochs, val_loss,'r',label='Validation accuaricy')

plt.title('Training and Validation lost')

plt.legend()



plt.show()
random.shuffle(test_imgs)

X_test, y_test = read_and_process_image(test_imgs[:10])

x = np.array(X_test)

test_datagen = ImageDataGenerator(rescale=1./255)
i=0

text_labels = []

plt.figure(figsize=(30,20))

for batch in test_datagen.flow(x,batch_size=1):

    pred=model.predict(batch)

    if pred>0.9:

        text_labels.append('duck')

    else:

        text_labels.append('road')

    plt.subplot(5/columns+1,columns, i+1)

    plt.title('This is a '+text_labels[i])

    imgplot=plt.imshow(batch[0])

    i+=1

    if i%10 ==0:

        break

plt.show()
dir_t='../input/testok/twosmall/small'

test_two =['../input/testok/twosmall/small/{}'.format(i) for i in os.listdir(dir_t) if 'two' in i]



random.shuffle(test_two)
X_test, y_test = read_and_process_image(test_two[:5])

x = np.array(X_test)

test_datagen = ImageDataGenerator(rescale=1./255)
i=0

text_labels = []

plt.figure(figsize=(30,20))

for batch in test_datagen.flow(x,batch_size=1):

    pred=model.predict(batch)

    if pred>0.5:

        text_labels.append('duck found!')

    else:

        text_labels.append('no duck ')

    plt.subplot(5/columns+1,columns, i+1)

    plt.title(''+text_labels[i])

    imgplot=plt.imshow(batch[0])

    i+=1

    if i%10 ==0:

        break

plt.show()