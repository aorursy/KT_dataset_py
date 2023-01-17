# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

PATH="../input/rsna-pneumonia-detection-challenge"

print(os.listdir(PATH))
class_info_df = pd.read_csv(PATH+'/stage_2_detailed_class_info.csv')

train_labels_df = pd.read_csv(PATH+'/stage_2_train_labels.csv')
class_info_df.sample(5)
train_labels_df.sample(5)
print("Detailed class info -  rows:",class_info_df.shape[0]," columns:", class_info_df.shape[1])

print("Train labels -  rows:",train_labels_df.shape[0]," columns:", train_labels_df.shape[1])
import pandas as pd 

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from matplotlib.patches import Rectangle

import seaborn as sns

import pydicom as dcm

%matplotlib inline 

IS_LOCAL = False

import os
f, ax = plt.subplots(1,1, figsize=(6,4))

total = float(len(class_info_df))

sns.countplot(class_info_df['class'],order = class_info_df['class'].value_counts().index, palette='Set3')

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(100*height/total),

            ha="center") 

plt.show()
train_class_df = train_labels_df.merge(class_info_df, left_on='patientId', right_on='patientId', how='inner')

train_class_df.sample(20)

image_sample_path_train = os.listdir(PATH+'/stage_2_train_images')

print(image_sample_path_train)
image_sample_path_test= os.listdir(PATH+'/stage_2_test_images')

print(image_sample_path_test)
image_train_path = os.listdir(PATH+'/stage_2_train_images')

image_test_path = os.listdir(PATH+'/stage_2_test_images')

print("Number of images in train set:", len(image_train_path),"\nNumber of images in test set:", len(image_test_path))
print("Unique patientId in  train_class_df: ", train_class_df['patientId'].nunique())      

samplePatientID = list(train_class_df[:3].T.to_dict().values())[0]['patientId']

samplePatientID = samplePatientID+'.dcm'

dicom_file_path = os.path.join(PATH,"stage_2_train_images/",samplePatientID)

dicom_file_dataset = dcm.read_file(dicom_file_path)

dicom_file_dataset
def show_dicom_images(data):

    img_data = list(data.T.to_dict().values())

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(img_data):

        patientImage = data_row['patientId']+'.dcm'

        imagePath = os.path.join(PATH,"stage_2_train_images/",patientImage)

        data_row_img_data = dcm.read_file(imagePath)

        modality = data_row_img_data.Modality

        age = data_row_img_data.PatientAge

        sex = data_row_img_data.PatientSex

        data_row_img = dcm.dcmread(imagePath)

        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}\nWindow: {}:{}:{}:{}'.format(

                data_row['patientId'],

                modality, age, sex, data_row['Target'], data_row['class'], 

                data_row['x'],data_row['y'],data_row['width'],data_row['height']))

    plt.show()

show_dicom_images(train_class_df[train_class_df['Target']==1].sample(9))

show_dicom_images(train_class_df[train_class_df['Target']==0].sample(9))
def show_dicom_images_with_boxes(data):

    img_data = list(data.T.to_dict().values())

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(img_data):

        patientImage = data_row['patientId']+'.dcm'

        imagePath = os.path.join(PATH,"stage_2_train_images/",patientImage)

        data_row_img_data = dcm.read_file(imagePath)

        modality = data_row_img_data.Modality

        age = data_row_img_data.PatientAge

        sex = data_row_img_data.PatientSex

        data_row_img = dcm.dcmread(imagePath)

        ax[i//3, i%3].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title('ID: {}\nModality: {} Age: {} Sex: {} Target: {}\nClass: {}'.format(

                data_row['patientId'],modality, age, sex, data_row['Target'], data_row['class']))

        rows = train_class_df[train_class_df['patientId']==data_row['patientId']]

        box_data = list(rows.T.to_dict().values())

        for j, row in enumerate(box_data):

            ax[i//3, i%3].add_patch(Rectangle(xy=(row['x'], row['y']),

                        width=row['width'],height=row['height'], 

                        color="yellow",alpha = 0.1))   

    plt.show()
show_dicom_images_with_boxes(train_class_df[train_class_df['Target']==1]).sample(9)

! pip install tensorflow --upgrade

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import os

import numpy as np

import matplotlib.pyplot as plt
train_dataa = train_class_df[train_class_df['Target']==1]

train_dataa


batch_size = 128

epochs = 15

IMG_HEIGHT = 150

IMG_WIDTH = 150
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data

test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,

                                                           directory= image_train_path,

                                                           shuffle=True,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           class_mode='binary')
test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,

                                                           directory= image_test_path,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           class_mode='binary')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



import tensorflow as tf



model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1, activation='sigmoid')

])
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

from sklearn.model_selection import train_test_split



tr_data, val_data = train_test_split(image_sample_path_train)
tr_data
len(val_data)
# history = model.fit(tr_data, epochs=20,

#                     validation_data=val_data)
import csv

# empty dictionary

pneumonia_locations = {}

# load table

with open(os.path.join('/kaggle/input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'), mode='r') as infile:

    # open reader

    reader = csv.reader(infile)

    # skip header

    next(reader, None)

    # loop through rows

    for rows in reader:

        # retrieve information

        filename = rows[0]

        location = rows[1:5]

        pneumonia = rows[5]

        # if row contains pneumonia add label to dictionary

        # which contains a list of pneumonia locations per filename

        if pneumonia == '1':

            # convert string to float to int

            location = [int(float(i)) for i in location]

            # save pneumonia location in dictionary

            if filename in pneumonia_locations:

                pneumonia_locations[filename].append(location)

            else:

                pneumonia_locations[filename] = [location]

from tensorflow import keras

class generator(keras.utils.Sequence):

    

    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=256, shuffle=True, augment=False, predict=False):

        self.folder = folder

        self.filenames = filenames

        self.pneumonia_locations = pneumonia_locations

        self.batch_size = batch_size

        self.image_size = image_size

        self.shuffle = shuffle

        self.augment = augment

        self.predict = predict

        self.on_epoch_end()

        

    def __load__(self, filename):

        # load dicom file as numpy array

        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array

        # create empty mask

        msk = np.zeros(img.shape)

        # get filename without extension

        filename = filename.split('.')[0]

        # if image contains pneumonia

        if filename in self.pneumonia_locations:

            # loop through pneumonia

            for location in self.pneumonia_locations[filename]:

                # add 1's at the location of the pneumonia

                x, y, w, h = location

                msk[y:y+h, x:x+w] = 1

        # resize both image and mask

        img = resize(img, (self.image_size, self.image_size), mode='reflect')

        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5

        # if augment then horizontal flip half the time

        if self.augment and random.random() > 0.5:

            img = np.fliplr(img)

            msk = np.fliplr(msk)

        # add trailing channel dimension

        img = np.expand_dims(img, -1)

        msk = np.expand_dims(msk, -1)

        return img, msk

    

    def __loadpredict__(self, filename):

        # load dicom file as numpy array

        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array

        # resize image

        img = resize(img, (self.image_size, self.image_size), mode='reflect')

        # add trailing channel dimension

        img = np.expand_dims(img, -1)

        return img

        

    def __getitem__(self, index):

        # select batch

        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]

        # predict mode: return images and filenames

        if self.predict:

            # load files

            imgs = [self.__loadpredict__(filename) for filename in filenames]

            # create numpy batch

            imgs = np.array(imgs)

            return imgs, filenames

        # train mode: return images and masks

        else:

            # load files

            items = [self.__load__(filename) for filename in filenames]

            # unzip images and masks

            imgs, msks = zip(*items)

            # create numpy batch

            imgs = np.array(imgs)

            msks = np.array(msks)

            return imgs, msks

        

    def on_epoch_end(self):

        if self.shuffle:

            random.shuffle(self.filenames)

        

    def __len__(self):

        if self.predict:

            # return everything

            return int(np.ceil(len(self.filenames) / self.batch_size))

        else:

            # return full batches only

            return int(len(self.filenames) / self.batch_size)
import random



# load and shuffle filenames

folder = PATH+'/stage_2_train_images'

filenames = os.listdir(folder)

random.shuffle(filenames)

# split into train and validation filenames

n_valid_samples = 2560

train_filenames = filenames[n_valid_samples:]

valid_filenames = filenames[:n_valid_samples]

print('n train samples', len(train_filenames))

print('n valid samples', len(valid_filenames))

n_train_samples = len(filenames) - n_valid_samples
def create_downsample(channels, inputs):

    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)

    x = keras.layers.MaxPool2D(2)(x)

    return x



def create_resblock(channels, inputs):

    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)

    x = keras.layers.BatchNormalization(momentum=0.9)(x)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)

    return keras.layers.add([x, inputs])



def create_network(input_size, channels, n_blocks=2, depth=4):

    # input

    inputs = keras.Input(shape=(input_size, input_size, 1))

    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)

    # residual blocks

    for d in range(depth):

        channels = channels * 2

        x = create_downsample(channels, x)

        for b in range(n_blocks):

            x = create_resblock(channels, x)

    # output

    x = keras.layers.BatchNormalization(momentum=0.9)(x)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    outputs = keras.layers.UpSampling2D(2**depth)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

import pydicom

from skimage import io

from skimage import measure

from skimage.transform import resize

# define iou or jaccard loss function

def iou_loss(y_true, y_pred):

    y_true = tf.reshape(y_true, [-1])

    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)

    return 1 - score



# combine bce loss and iou loss

def iou_bce_loss(y_true, y_pred):

    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)



# mean iou as a metric

def mean_iou(y_true, y_pred):

    y_pred = tf.round(y_pred)

    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])

    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    smooth = tf.ones(tf.shape(intersect))

    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))



# create network and compiler

model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)

model.compile(optimizer='adam',

              loss=iou_bce_loss,

              metrics=['accuracy', mean_iou])



# cosine learning rate annealing

def cosine_annealing(x):

    lr = 0.001

    epochs = 25

    return lr*(np.cos(np.pi*x/epochs)+1.)/2

learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)



# create train and validation generators

folder = PATH+'/stage_2_train_images'

train_gen = generator(folder, train_filenames, pneumonia_locations, batch_size=32, image_size=256, shuffle=True, augment=True, predict=False)

valid_gen = generator(folder, valid_filenames, pneumonia_locations, batch_size=32, image_size=256, shuffle=False, predict=False)



history = model.fit_generator(train_gen, validation_data=valid_gen, callbacks=[learning_rate], epochs=5, workers=4, use_multiprocessing=True)
