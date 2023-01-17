import os

import cv2

import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train_dir = '../input/train/'

test_dir = '../input/test/'
train_images = [train_dir + i for i in os.listdir(train_dir)]

train_cats = [train_dir + i for i in os.listdir(train_dir) if 'cat' in i]

train_dogs = [train_dir + i for i in os.listdir(train_dir) if 'dog' in i]



test_images =  [test_dir + i for i in os.listdir(test_dir)]
print('the number of total train images:',len(train_images))

print('the number of train cats images:',len(train_cats))

print('the number of train dogs images:',len(train_dogs))

print('the number of total test images:',len(test_images))
random.seed(100)



original_train_images = train_dogs[:12000] + train_cats[:12000]



evaluation_images = train_dogs[12000:12500] + train_cats[12000:12500]



random.shuffle(evaluation_images)

random.shuffle(original_train_images)



section = int(len(original_train_images) * 0.75)



train_images = original_train_images[:section]

validation_images = original_train_images[section:]
print(len(train_images))

print(len(validation_images))
# imgsize = 150

# channels = 3



# def read_images(one_img):

#     img = cv2.imread(one_img,cv2.IMREAD_ANYCOLOR)

#     img_arr = cv2.resize(img,(imgsize,imgsize),interpolation=cv2.INTER_CUBIC)

#     img_arr = img_arr / 255.0

#     return img_arr
# im = read_images(tr_images[0])

# plt.imshow(im)


# def pre_data(images):

#     lens = len(images)

#     data = np.ndarray((lens,imgsize,imgsize,channels), dtype=np.uint8)

    

#     for i, img_file in enumerate(images):

#         image = read_images(img_file)

#         label = np.where('dog' in tr_images[i],1,0)

#         data[i] = image

        

#     return data
from keras.preprocessing import image



imgsize = 150

channels = 3



def prep_data(images):

    count = len(images)

    X = np.ndarray((count, imgsize, imgsize, channels), dtype=np.float32)

    y = np.zeros((count,), dtype=np.float32)

    

    for i, image_file in enumerate(images):

        img = image.load_img(image_file, target_size=(imgsize, imgsize))

        X[i] = image.img_to_array(img)

        if 'dog' in image_file:

            y[i] = 1.

        if i%1000 == 0: print('Processed {} of {}'.format(i, count))

    

    return X, y
X_train, y_train = prep_data(train_images)
print("Train shape: ",X_train.shape)

print("Train shape: ",y_train.shape)
X_validation, y_validation = prep_data(validation_images)
train_datagen = image.ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,)



validation_datagen = image.ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 128



train_generator = train_datagen.flow(

    X_train,

    y_train,

    batch_size=BATCH_SIZE)



validation_generator = validation_datagen.flow(

    X_validation,

    y_validation,

    batch_size=BATCH_SIZE)
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten

from keras.optimizers import RMSprop

from keras.models import Sequential



model = Sequential()



model.add(Conv2D(32,(3,3),activation = 'relu',input_shape = (imgsize,imgsize,channels)))

model.add(MaxPooling2D((2,2)))





model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# model = Sequential()



# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=train.shape[1:]))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.5))



# model.add(Conv2D(64, (3, 3), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.5))



# model.add(Conv2D(128, (3, 3), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.5))



# model.add(Conv2D(256, (3, 3), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.5))



# model.add(Flatten())

# model.add(Dense(512, activation='relu'))

# model.add(Dropout(0.5))



# model.add(Dense(1, activation='sigmoid'))



# model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
train_steps = len(train_images)/BATCH_SIZE

validation_steps = len(validation_images)/BATCH_SIZE



history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_steps,

    epochs=100,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    verbose=1)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label = 'Training acc')

plt.plot(epochs,val_acc,'b',label = 'Validation acc')

plt.title('Training and Validation accuracy')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend()



plt.figure()



epochs = range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label = 'Training loss')

plt.plot(epochs,val_loss,'b',label = 'Validation loss')

plt.title('Training and Validation loss')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend()



plt.show()
model.save('dogs-v-cat-data-1.h5')
import json

with open('dogs-v-cat-data-1.h5-history.json', 'w') as f:

    json.dump(history.history, f)
X_evaluation, y_evaluation = prep_data(evaluation_images)

X_evaluation /= 255
evaluation = model.evaluate(X_evaluation, y_evaluation)
evaluation
X_test, _ = prep_data(test_images)

X_test /= 255.
predictions = model.predict(X_test)
for i in range(0,10):

    if predictions[i, 0] >= 0.5: 

        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))

    else: 

        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))

        

    plt.imshow(image.array_to_img(X_test[i]))

    plt.show()