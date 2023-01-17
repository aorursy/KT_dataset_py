import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from numpy import expand_dims

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split

import gc

import tensorflow as tf

from tensorflow.keras import layers, models

from sklearn.metrics import auc

from sklearn.metrics import roc_curve
os.listdir('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database')
#Exploring the dataset directory in more depth

#Run just in case you want to know the images are segmented in each folder



#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))
normal_quantity = 0

vpneumonia_quantity = 0

covid_quantity = 0



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.png'):

            if 'NORMAL' in filename:

                normal_quantity += 1

            if 'Viral' in filename:

                vpneumonia_quantity += 1

            if 'COVID' in filename:

                covid_quantity += 1



print('Normal category quantity: ',normal_quantity)

print('Viral Pneumonia category quantity: ',vpneumonia_quantity)

print('COVID category quantity: ',covid_quantity)
%matplotlib inline



plt.figure(figsize=(30,20))



ax1 = plt.subplot(131)

plt.imshow(mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/COVID-19 (3).png'))



ax2 = plt.subplot(132)

plt.imshow(mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/COVID-19 (27).png'))



ax3 = plt.subplot(133)

plt.imshow(mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/COVID-19 (30).png'))



plt.show()
%matplotlib inline



plt.figure(figsize=(30,20))



ax1 = plt.subplot(131)

plt.imshow(mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia/Viral Pneumonia (49).png'))



ax2 = plt.subplot(132)

plt.imshow(mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia/Viral Pneumonia (82).png'))



ax3 = plt.subplot(133)

plt.imshow(mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/Viral Pneumonia/Viral Pneumonia (1114).png'))



plt.show()
#Let's plot some normal images randomly to know how they look like, so we can determine important factors as color.



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



plt.figure(figsize=(30,20))



ax1 = plt.subplot(131)

#cv2.imread('data/src/lena.jpg')

plt.imshow(mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL/NORMAL (764).png'))



ax2 = plt.subplot(132)

plt.imshow(mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL/NORMAL (277).png'))



ax3 = plt.subplot(133)

plt.imshow(mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL/NORMAL (121).png'))



plt.show()
#Let's test this hypothesis:

print('Shape of image array using cv2.imread: ',cv2.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL/NORMAL (764).png').shape)

print('Shape of image array using mpimg.imread: ',mpimg.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL/NORMAL (764).png').shape)
normal_shape = []

vpneumonia_shape = []

covid_shape = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.png'):

            if 'NORMAL' in filename:

                normal_shape.append(cv2.imread(os.path.join(dirname, filename)).shape)

            if 'Viral' in filename:

                vpneumonia_shape.append(cv2.imread(os.path.join(dirname, filename)).shape)

            if 'COVID' in filename:

                covid_shape.append(cv2.imread(os.path.join(dirname, filename)).shape)

print('Unique shapes at Normal images set: ',set(normal_shape))

print('Unique shapes at Viral Pneumonia images set: ',set(vpneumonia_shape))

print('Unique shapes at COVID images set: ',set(covid_shape))
%matplotlib inline

plt.figure()

image = cv2.imread('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL/NORMAL (121).png')

image = cv2.resize(image, (128, 128))

plt.imshow(image)

plt.show()      
def data_augmentation(dirname,filename):

    

    """

    This function will perform data augmentation: 

    for each one of the images, will create shifted, expanded/reduced, darker/lighter, rotated images. 9 for every modification type. 

    In total, we will create 36 extra images for every one in the original dataset.

    """

    

    image_data = []

    #reading the image

    image = cv2.imread(os.path.join(dirname, filename))

    image = cv2.resize(image, (128, 128))

    #expanding the image dimension to one sample

    samples = expand_dims(image, 0)

    # creating the image data augmentation generators

    datagen1 = ImageDataGenerator(width_shift_range=[-100,100])

    datagen2 = ImageDataGenerator(zoom_range=[0.7,1.0])

    datagen3 = ImageDataGenerator(brightness_range=[0.2,1.0])

    datagen4 = ImageDataGenerator(rotation_range=25)

      

    # preparing iterators

    it1 = datagen1.flow(samples, batch_size=1)

    it2 = datagen2.flow(samples, batch_size=1)

    it3 = datagen3.flow(samples, batch_size=1)

    it4 = datagen4.flow(samples, batch_size=1)

    image_data.append(image)

    for i in range(9):

        # generating batch of images

        batch1 = it1.next()

        batch2 = it2.next()

        batch3 = it3.next()

        batch4 = it4.next()

        # convert to unsigned integers

        image1 = batch1[0].astype('uint8')

        image2 = batch2[0].astype('uint8')

        image3 = batch3[0].astype('uint8')

        image4 = batch4[0].astype('uint8')

        #appending to the list of images

        image_data.append(image1)

        image_data.append(image2)

        image_data.append(image3)

        image_data.append(image4)

        

    return image_data
#Let's test our function.

result = data_augmentation('/kaggle/input/covid19-radiography-database/COVID-19 Radiography Database/NORMAL','NORMAL (121).png')
#Let's plot an image

%matplotlib inline

plt.figure()

image = result[6]

plt.imshow(image)

plt.show()  
def data_transformation(keyword):

    

    """

    This function receives a keyword as parameter to determine the kind of set it's going to process.

    It uses data_augmentation function to expand the image quantity, then resizes the images and finally returns a list containing the images already processed.

    IMPORTANT: Maybe you'll notice we don't use any Keras method which could make easier the image processing. Instead, we decided to process the images using our own functions.

    """

    

    images = []

    counter = 0

    

    if keyword == 'NORMAL':

        for dirname, _, filenames in os.walk('/kaggle/input'):

            for filename in filenames:

                if (filename.endswith('.png')) and ('NORMAL' in filename):

                    image = cv2.imread(os.path.join(dirname, filename))

                    image = cv2.resize(image, (128, 128))

                    images.append(image)

                    counter += 1

        for dirname, _, filenames in os.walk('/kaggle/input'):

            for filename in filenames:

                if (filename.endswith('.png')) and ('NORMAL' in filename):

                    result = data_augmentation(dirname,filename)

                    for i in range(len(result)):

                        if i==0:

                            continue

                        else:

                            images.append(result[i])

                            counter += 1

                if counter >= 8200:

                    break

            if counter >= 8200:

                break

                

    if keyword == 'Viral':

        for dirname, _, filenames in os.walk('/kaggle/input'):

            for filename in filenames:

                if (filename.endswith('.png')) and ('Viral' in filename):

                    image = cv2.imread(os.path.join(dirname, filename))

                    image = cv2.resize(image, (128, 128))

                    images.append(image)

                    counter += 1

        for dirname, _, filenames in os.walk('/kaggle/input'):

            for filename in filenames:

                if (filename.endswith('.png')) and ('Viral' in filename):

                    result = data_augmentation(dirname,filename)

                    for i in range(len(result)):

                        if i==0:

                            continue

                        else:

                            images.append(result[i])

                            counter += 1

                if counter >= 8200:

                    break

            if counter >= 8200:

                break

                                

    if keyword == 'COVID':

        for dirname, _, filenames in os.walk('/kaggle/input'):

            for filename in filenames:

                if (filename.endswith('.png')) and ('COVID' in filename):

                    result = data_augmentation(dirname,filename)

                    for i in range(len(result)):

                        images.append(result[i])

    return images
normal = data_transformation('NORMAL')[:5000]

viral_pneumonia = data_transformation('Viral')[:5000]

covid = data_transformation('COVID')[:5000]
#Class combination

X = normal + viral_pneumonia + covid

len(X)
#Transforming from list to numpy array.

X = np.array(X)

X.shape
class_names = ['Normal','Viral Pneumonia','COVID-19']
#Creating labels.

y = []

for i in range(5000):

    y.append(0)

for i in range(5000):

    y.append(1)

for i in range(5000):

    y.append(2)

y = np.array(y)

len(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0,shuffle=True)
#del normal_shape

#del vpneumonia_shape

#del covid_shape

del X

#del image

del y

del normal

del viral_pneumonia

del covid

gc.collect()
#X_train, X_test = X_train / 255.0, X_test / 255.0
gc.collect()
# Detect and init the TPU

#tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

#tf.config.experimental_connect_to_cluster(tpu)

#tf.tpu.experimental.initialize_tpu_system(tpu)
# Instantiate a distribution strategy

#tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
# instantiating the model in the strategy scope creates the model on the TPU

#with tpu_strategy.scope():

    

#    model = models.Sequential()

#    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))

#    model.add(layers.MaxPooling2D((2, 2)))

#    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

#    model.add(layers.MaxPooling2D((2, 2)))

#    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

#    model.add(layers.Flatten())

#    model.add(layers.Dense(256, activation='relu'))

#    model.add(layers.Dense(3))

    

#    model.compile(optimizer='adam',

#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

    

#history = model.fit(X_train, y_train, epochs=6,validation_data=(X_test, y_test))
model = models.Sequential()

model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(3))
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#Above option if you want to improve accuracy. You could also focus on improving precision or if you want to go beyond,

#educe false positive cases optimizing AUC function.

#Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
gc.collect()
history = model.fit(X_train, y_train, epochs=6,validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('Model accuracy: ',test_acc)
y_pred = model.predict(X_test)
class_names[np.argmax(y_pred[1])]
class_names[y_test[1]]
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
import seaborn as sns

conf_matrix = pd.DataFrame(matrix, index = ['Normal','Viral Pneumonia','COVID-19'],columns = ['Normal','Viral Pneumonia','COVID-19'])

#Normalizing

conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize = (15,15))

sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})
#Saving the model



!mkdir -p saved_model

model.save('saved_model/model') 