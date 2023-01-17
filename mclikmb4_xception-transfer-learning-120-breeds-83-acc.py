!pip install tf_explain

#!pip install split-folders

#!conda install -y gdown
import os

import pandas as pd



import xml.etree.ElementTree as ET

#import gdown

import time

import math

import cv2

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import matplotlib.image as image

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



from keras.applications.xception import Xception, preprocess_input

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential



from keras.utils import np_utils

from keras.utils import Sequence

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tf_explain.core.activations import ExtractActivations



from tensorflow.keras.applications.xception import decode_predictions

%matplotlib inline

from sklearn.metrics import classification_report



from PIL import Image

import matplotlib.image as mpimg

from imgaug import augmenters as iaa

print("Loaded all libraries")
image_path = '../input/stanford-dogs-dataset/images/Images'

#image_path ='/media/marco/DATA/OC_Machine_learning/section_6/DATA/Images/'

num_of_categories = 120

image_size = 299

batch_size = 16
breed_list = sorted(os.listdir(image_path))



num_classes = len(breed_list)

print("{} breeds".format(num_classes))
# Define a time counter function to test the algorythms performance 

_start_time = time.time()



def process_time_starts():

    global _start_time 

    _start_time = time.time()



def time_elapsed():

    t_sec = round(time.time() - _start_time)

    (t_min, t_sec) = divmod(t_sec,60)

    (t_hour,t_min) = divmod(t_min,60) 

    print('The process took: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))




# copy from https://www.kaggle.com/gabrielloye/dogs-inception-pytorch-implementation

# reduce the background noise



os.mkdir('data')

for breed in breed_list:

    os.mkdir('data/' + breed)

print('Created {} folders to store cropped images of the different breeds.'.format(len(os.listdir('data'))))



%%time

for breed in os.listdir('data'):

    for file in os.listdir('../input/stanford-dogs-dataset/annotations/Annotation/{}'.format(breed)):

        img = Image.open('../input/stanford-dogs-dataset/images/Images/{}/{}.jpg'.format(breed, file))

        tree = ET.parse('../input/stanford-dogs-dataset/annotations/Annotation/{}/{}'.format(breed, file))

        xmin = int(tree.getroot().findall('object')[0].find('bndbox').find('xmin').text)

        xmax = int(tree.getroot().findall('object')[0].find('bndbox').find('xmax').text)

        ymin = int(tree.getroot().findall('object')[0].find('bndbox').find('ymin').text)

        ymax = int(tree.getroot().findall('object')[0].find('bndbox').find('ymax').text)

        img = img.crop((xmin, ymin, xmax, ymax))

        img = img.convert('RGB')

        img = img.resize((image_size, image_size))

        img.save('data/' + breed + '/' + file + '.jpg')
plt.figure(figsize=(10, 10))

for i in range(9):

    plt.subplot(331 + i) # showing 9 random images

    breed = np.random.choice(breed_list) # random breed

    dog = np.random.choice(os.listdir('../input/stanford-dogs-dataset/annotations/Annotation/' + breed)) # random image 

    img = Image.open('../input/stanford-dogs-dataset/images/Images/' + breed + '/' + dog + '.jpg') 

    tree = ET.parse('../input/stanford-dogs-dataset/annotations/Annotation/' + breed + '/' + dog) # init parser for file given

    root = tree.getroot() # idk what's it but it's from documentation

    objects = root.findall('object') # finding all dogs. An array

    plt.imshow(img) # displays photo

    for o in objects:

        bndbox = o.find('bndbox') # reading border coordinates

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]) # showing border

        plt.text(xmin, ymin, o.find('name').text, bbox={'ec': None}) # printing breed
label_maps = {}

label_maps_rev = {}

for i, v in enumerate(breed_list):

    label_maps.update({v: i})

    label_maps_rev.update({i : v})
def paths_and_labels():

    paths = list()

    labels = list()

    targets = list()

    for breed in breed_list:

        base_name = "./data/{}/".format(breed)

        for img_name in os.listdir(base_name):

            paths.append(base_name + img_name)

            labels.append(breed)

            targets.append(label_maps[breed])

    return paths, labels, targets



paths, labels, targets = paths_and_labels()



assert len(paths) == len(labels)

assert len(paths) == len(targets)



targets = np_utils.to_categorical(targets, num_classes=num_classes)
class ImageGenerator(Sequence):

    

    def __init__(self, paths, targets, batch_size, shape, augment=False):

        self.paths = paths

        self.targets = targets

        self.batch_size = batch_size

        self.shape = shape

        self.augment = augment

        

    def __len__(self):

        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    

    def __getitem__(self, idx):

        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)

        y = np.zeros((self.batch_size, num_classes, 1))

        for i, path in enumerate(batch_paths):

            x[i] = self.__load_image(path)

        y = self.targets[idx * self.batch_size : (idx + 1) * self.batch_size]

        return x, y

    

    def __iter__(self):

        for item in (self[i] for i in range(len(self))):

            yield item

            

    def __load_image(self, path):

        image = cv2.imread(path)

        image = preprocess_input(image)

        if self.augment:

            seq = iaa.Sequential([

                iaa.OneOf([

                    iaa.Fliplr(0.5),

                    iaa.Flipud(0.5),

                    iaa.Sometimes(0.5,

                    

                    ),

                    iaa.Affine(

                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},

                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},

                        rotate=(-40, 40),

                        shear=(-8, 8)

                    )

                ])

            ], random_order=True)

            image = seq.augment_image(image)

        return image
x_train, x_test, y_train, y_test = train_test_split(paths, targets, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)



train_ds = ImageGenerator(x_train, y_train, batch_size=32, shape=(image_size, image_size,3), augment=True)

val_ds = ImageGenerator(x_test, y_test, batch_size=32, shape=(image_size, image_size,3), augment=False)

test_ds = ImageGenerator(x_test, y_test, batch_size=32, shape=(image_size, image_size,3), augment=False)




#url = 'https://drive.google.com/uc?id=1aCFGR5c7Ap4JPzryR_RSgRYc7FRYgbyG'



#output = 'xception_weights.h5'



#gdown.download(url, output, quiet=False)
base_model = tf.keras.applications.xception.Xception(weights='imagenet',include_top=False, pooling='avg')#Summary of Xception Model



base_model.trainable = False





#pre_trained_model.summary()



flat_dim = 5 * 5 * 2048



my_model = Sequential(base_model)



#my_model.add(Flatten())

#my_model.add(Dropout(0.1)) # dropout added

my_model.add(Dense(1032, activation='relu',input_dim=flat_dim))

my_model.add(Dense(512, activation='relu'))

#my_model.add(Dropout(0.1))

my_model.add(Dense(256, activation='relu'))

my_model.add(Dense(120, activation='softmax'))



###################

total_epoch = 8

learning_rate_init = 0.00001

###################



def lr_scheduler(epoch):

    epoch += 1

   

    if epoch == 1:

        return learning_rate_init

    

    elif epoch >= 2 and epoch <= 40:

        return (0.2*epoch**3)*math.exp(-0.45*epoch)*learning_rate_init

    

    else:

        return lr_scheduler(40-1)

    



stage = [i for i in range(0,25)]

learning_rate = [lr_scheduler(x) for x in stage]

plt.plot(stage, learning_rate)

print(learning_rate)
# Callbacks



scheduler = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

early_stop = EarlyStopping(monitor='val_accuracy', patience = 6, mode='max', min_delta=1, verbose=1)
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
process_time_starts()



hist = my_model.fit_generator(generator=train_ds, steps_per_epoch=400, validation_data=val_ds,  validation_steps=90, epochs=8, callbacks=[scheduler])



                  
          



time_elapsed()
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax = ax.ravel()



for i, met in enumerate(['accuracy', 'loss']):

    ax[i].plot(hist.history[met])

    ax[i].plot(hist.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
my_model.save('my_model.h5', overwrite=True) 

my_model.save_weights('dog_breed_xcept_weights.h5', overwrite=True)

print("Saved model to disk")
test_loss, test_accuracy = my_model.evaluate_generator(generator=test_ds,steps=int(100))



print("Test results \n Loss:",test_loss,'\n Accuracy',test_accuracy)


#report = classification_report(test_ds.classes, pred, target_names=class_to_id)

#print(report)
def download_and_predict(url, filename):

    # download and save

    os.system("curl -s {} -o {}".format(url, filename))

    img = Image.open(filename)

    img = img.convert('RGB')

    img = img.resize((299, 299))

    img.save(filename)

    # show image

    plt.figure(figsize=(4, 4))

    plt.imshow(img)

    plt.axis('off')

    # predict

    img = image.imread(filename)

    img = preprocess_input(img)

    probs = my_model.predict(np.expand_dims(img, axis=0))

    for idx in probs.argsort()[0][::-1][:5]:

        print("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])
download_and_predict("https://cdn.pixabay.com/photo/2018/08/12/02/52/belgian-mallinois-3599991_1280.jpg",

                     "test_1.jpg")



download_and_predict("http://giandonet.altervista.org/Marco/ala.JPG",

                     "test_2.jpg")

download_and_predict("http://giandonet.altervista.org/Marco/surfingdog.jpg",

                     "test_3.jpg")