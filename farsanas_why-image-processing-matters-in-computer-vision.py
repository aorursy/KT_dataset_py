#Video link

from IPython.display import YouTubeVideo      

YouTubeVideo('u7bE0N0WB1A')
# load cifar10 data from keras "https://www.cs.toronto.edu/~kriz/cifar.html"



from tensorflow.keras.datasets import cifar10
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np



from tensorflow.keras.preprocessing.image import ImageDataGenerator
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train.shape
#Video link

from IPython.display import YouTubeVideo      

YouTubeVideo('lxLyP2yKlp8')
# Displaying original images

plt.figure(figsize=(15,15))

for i in range(0,9):

    plt.subplot(330+1+i)

    image = X_train[i]

    plt.imshow(image.astype('uint8'))
datagen = ImageDataGenerator()



# fit parameters from data

datagen.fit(X_train)



for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):

    for i in range(0, 9):

        plt.subplot(330 + 1 + i)

        plt.imshow(X_batch[i].astype('uint8'))

    break
training_generator = datagen.flow_from_directory(

        '../input/multiclass-dataset/dataset',

        color_mode="rgb",

        batch_size=32,

        class_mode="categorical",                                  # This has more than 2 classes

        shuffle=True,

        seed=42,

        target_size=(100,100))
Xbatch, Ybatch = training_generator.next()
#Total shape of images

Xbatch.shape
#Shape of labels of imges

Ybatch.shape
Ybatch[:5,]
Xbatch[0].shape


i=0

plt.figure(figsize=(15,15))



for img in Xbatch:

    plt.subplot(5,6, i+1)

    plt.tight_layout()

    plt.imshow(img.astype('uint8'))

    i=i+1    
gray_training_generator = datagen.flow_from_directory(

        '../input/multiclass-dataset/dataset',

        color_mode="grayscale",

        batch_size=32,

        class_mode="categorical",                                  # This has more than 2 classes

        shuffle=True,

        seed=42,

        target_size=(100, 100))
XbatchGray, YbatchGray = gray_training_generator.next()
batch_size = 9

i=0

plt.figure(figsize=(10,10))



for img in XbatchGray:

    plt.subplot(5,6, i+1)

    plt.tight_layout()

    gray_img = img[:,:,0]

    plt.imshow(gray_img.astype('uint8'), cmap='gray')

    i=i+1  
batch_size = 8

datagen = ImageDataGenerator(rescale= 1. / 255.)

datagen.fit(Xbatch)

i=0

for img_batch in datagen.flow(Xbatch, batch_size=9):

    for img in img_batch:

        plt.subplot(330 + 1 + i)

        plt.tight_layout()

        plt.imshow(img)

        i=i+1    

    if i >= batch_size:

        break
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

datagen.fit(Xbatch)

i=0

for img_batch in datagen.flow(Xbatch, batch_size=9):

    for img in img_batch:

        plt.subplot(330 + 1 + i)

        plt.tight_layout()

        plt.imshow(img.astype('uint8'))

        i=i+1    

    if i >= batch_size:

        break
datagen = ImageDataGenerator(rotation_range=90)

datagen.fit(Xbatch)



i=0

for img_batch in datagen.flow(Xbatch, batch_size=9):

    for img in img_batch:

        plt.subplot(330 + 1 + i)

        plt.tight_layout()

        plt.imshow(img.astype('uint8'))

        i=i+1    

    if i >= batch_size:

        break
shift = 0.2

datagen = ImageDataGenerator(width_shift_range=shift, fill_mode='wrap')

datagen.fit(Xbatch)

i=0

for img_batch in datagen.flow(Xbatch, batch_size=9):

    for img in img_batch:

        plt.subplot(330 + 1 + i)

        plt.tight_layout()

        plt.imshow(img.astype('uint8'))

        i=i+1    

    if i >= batch_size:

        break
shift = 0.2

datagen = ImageDataGenerator(height_shift_range=shift)

datagen.fit(Xbatch)

i=0

for img_batch in datagen.flow(Xbatch, batch_size=9):

    for img in img_batch:

        plt.subplot(330 + 1 + i)

        plt.tight_layout()

        plt.imshow(img.astype('uint8'))

        i=i+1    

    if i >= batch_size:

        break
datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

datagen.fit(Xbatch)

batch_size = 9

i=0

for img_batch in datagen.flow(Xbatch, batch_size=9):

    for img in img_batch:

        plt.subplot(330 + 1 + i)

        plt.tight_layout()

        plt.imshow(img.astype('uint8'))

        i=i+1    

    if i >= batch_size:

        break
datagen = ImageDataGenerator(zoom_range=0.75, fill_mode='constant')

datagen.fit(Xbatch)

batch_size = 9

i=0

for img_batch in datagen.flow(Xbatch, batch_size=9):

    for img in img_batch:

        plt.subplot(330 + 1 + i)

        plt.tight_layout()

        plt.imshow(img.astype('uint8'))

        i=i+1    

    if i >= batch_size:

        break
datagen = ImageDataGenerator(shear_range=130)

datagen.fit(Xbatch)

batch_size = 9

i=0

for img_batch in datagen.flow(Xbatch, batch_size=9):

    for img in img_batch:

        plt.subplot(330 + 1 + i)

        plt.tight_layout()

        plt.imshow(img.astype('uint8'))

        i=i+1    

    if i >= batch_size:

        break


#Video link

from IPython.display import YouTubeVideo      

YouTubeVideo('DbfpXtK4mLY')