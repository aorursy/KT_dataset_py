# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/ship/ship"))



# Any results you write to the current directory are saved as output.
import cv2   #use it in reading and resizing our Images.

import numpy as np  #process large, multi-dimensional arrays and matrices super easy and fast.

import pandas as pd #manipulating numerical tables and time series.

import matplotlib.pyplot as plt #plotting lines, bar-chart, graphs, histograms

%matplotlib inline 

#makes our plots appear in the notebook



import os #accessing your computer and file system.

import random # create random numbers, split or shuffle our data set.

import gc # garbage collector is an important tool for manually cleaning and deleting unnecessary variables.
#create a file path for both test and train sets

train_dir = '../input/ship/ship/train'

test_dir = '../input/ship/ship/val'



#create two variables train_dogs and train_cats

#write a list comprehension: os.listdir() to get all the images in the train data

#and retrieve all images with dog/cat in their name

train_ship = ['../input/ship/ship/train/{}'.format(i) for i in os.listdir(train_dir)if '1_' in i]

train_no_ship = ['../input/ship/ship/train/{}'.format(i) for i in os.listdir(train_dir) if '0_' in i]



#get our test images

test_imgs = ['../input/ship/ship/val/{}'.format(i) for i in os.listdir(test_dir)]



# with little computational power, we’re going to extract only 2000 images for both classes

train_imgs = train_ship + train_no_ship

random.shuffle(train_imgs) #randomly shuffle the train_imgs



# delete two columns to save memories

del train_ship

del train_no_ship

gc.collect()
#Import an image plotting module from matplotlib

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



#Run a for loop to plot the first three images in train_imgs

for ima in train_imgs[0:3]:

    img = mpimg.imread(ima)

#    print(img)

    imgplot = plt.imshow(img)

    plt.show()
#resize the images using the cv2 module

#declare the new dimensions we want to use: 150 by 150 for height and width and 3 channels



nrows = 150

ncolumns = 150

channels = 3 #change to 1 if you want to use grayscale image
def read_and_process_image(list_of_images):

    """

    Returns two arrays:

        X is an array of resized images

        y is an array of labels

    """

    X = [] #images

    y = [] #labels

    

    for image in list_of_images: #read images one after the other and resize them with the cv2 commands.

        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation = cv2.INTER_CUBIC)) #read the image

        #get the labels

        if '1_' in image:

            y.append(1)

        elif '0_' in image:

            y.append(0)

        

    return X, y

    
X, y = read_and_process_image(train_imgs)
X[0]
y
#We can’t plot the images in X with the mpimg module of matplotlib.image above 

#because these are now arrays of pixels not raw jpg files

#So we should use the imshow() command.





plt.figure(figsize = (20, 10))

columns = 5 

for i in range(columns):

    plt.subplot(5/ columns + 1, columns, i + 1)

    plt.imshow(X[i])
import seaborn as sns



#we delete the train_imgs, since it has already been converted to an array and saved in X.

del train_imgs

gc.collect()



#X and y are currently of type list (list of python array)

#convert list to numpy array so we can feed it into our model

X = np.array(X)

y = np.array(y)



#Lets plot the to be sure we just have two classes

#Plot a colorful diagram to confirm the number of classes

sns.countplot(y)

plt.title('Labels for Noships and ships')
#check the shape of data 

print("Shape of the image is:", X.shape)

print("Shape of the label is:", y.shape)



#keras model takes as input an array of (height, width,channels)
#split data into test and train set

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 2)



print("Shape of train images is", X_train.shape)

print("Shape of validation images is", X_val.shape)

print("Shape of train label is", y_train.shape)

print("Shape of validation label is", y_val.shape)
del X

del y

gc.collect()



#get the length of the train and validation data

ntrain = len(X_train)

nval = len(X_val)



# use batch size of 32

batch_size = 3



from keras import layers

from keras import models #Sequential model will be used

from keras import optimizers #contains different types of back propagation algorithm for training our model

from keras.preprocessing.image import ImageDataGenerator #(ImageDataGenerator) used when working with a small data set

from keras.preprocessing.image import img_to_array, load_img

#Keras comes prepackaged with many types of these pretrained models.

#We’ll be using the InceptionResNetV2

from keras.applications import InceptionResNetV2

conv_base = InceptionResNetV2(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))

#tell keras to download the model’s pretrained weights and save it in the variable conv_base.

#tell keras to fetch InceptionReNetV2 that was trained on the imagenet dataset.

#include_top = False: tells Keras not to download the fully connected layers of the pretrained model. 

conv_base.summary()
from keras import layers

from keras import models



model = models.Sequential()

model.add(conv_base) #we add our conv_base (InceptionResNetV2) to the model.

model.add(layers.Flatten()) #flatten the output from the conv_base because we want to pass it to our fully connected layer (classifier).

model.add(layers.Dense(256, activation = 'relu')) #added a Dense network with an output of 256 (number not fixed) 

model.add(layers.Dense(1, activation = 'sigmoid')) #the last layer has just 1 output. (Probability of classes)
model.summary()
print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False #freeze the conv_base and train only our own.

print('Number of trainable weights before freezing the conv base:', len(model.trainable_weights))
model.compile(loss ='binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-5), metrics = ['acc'])


train_datagen = ImageDataGenerator(rescale = 1./255, # scale the image between 0 and 1

                                  rotation_range = 0.2,

                                  width_shift_range = 0.2,

                                  height_shift_range = 0.2,

                                  shear_range = 0.2,

                                  zoom_range = 0.2,

                                  horizontal_flip = True,)



val_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow(X_train, y_train, batch_size = batch_size)

val_generator = val_datagen.flow(X_val, y_val, batch_size = batch_size)


history = model.fit_generator(train_generator, 

                             steps_per_epoch = ntrain // batch_size,

                             epochs = 40, 

                             validation_data = val_generator,

                             validation_steps = nval // batch_size)
model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')
#plot the train and val curve 

#get the details form the history object



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



#Train and validation accuracy

plt.plot(epochs, acc, 'b', label = 'Training accuracy')

plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')

plt.title('Training accuracy and Validation accuracy')

plt.legend()



plt.figure()



#Train and validation loss

plt.plot(epochs, loss, 'b', label = 'Training loss')

plt.plot(epochs, val_loss, 'r', label = 'Validation loss')

plt.title('Training loss and Validation loss')

plt.legend()



plt.show()
X_test, y_test = read_and_process_image(test_imgs[0:10])

x = np.array(X_test)

test_datagen = ImageDataGenerator(rescale = 1. /255)
i = 0 

text_labels = []

plt.figure(figsize = (30, 20))

for batch in test_datagen.flow(x, batch_size=1):

    pred =model.predict(batch)

    if pred > 0.5:

        text_labels.append('a ship')

    else:

        text_labels.append('not a ship')

    plt.subplot(5/ columns + 1, columns, i + 1)

    plt.title('This is ' + text_labels[i])

    implot = plt.imshow(batch[0])

    i += 1

    if i % 10 ==0:

        break

plt.show()